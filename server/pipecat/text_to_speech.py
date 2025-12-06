"""Text-to-Speech WebSocket server - テキスト入力から音声ストリーム出力

★ 設計:
- WebSocketでテキストメッセージを受信
- LLM → TTS のパイプラインで音声生成
- 音声はバイナリ（PCM int16）でストリーミング送信
- STT・VADレイヤーは不要
"""
import asyncio
import json
import time
import warnings
import os
from pathlib import Path
from typing import Optional

# NNPACK警告を抑制
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', message='.*NNPACK.*')

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from server.agent.segmenter import SentenceSegmenter
from server.agent.services.llm_service import LLMService
from server.agent.services.tts_service import TTSService


app = FastAPI(title="Text-to-Speech Agent")

STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# アクティブな接続を管理
active_connections: set[WebSocket] = set()


def np_int16_to_bytes(audio: np.ndarray) -> bytes:
    """int16 numpy array を bytes に変換"""
    return audio.astype(np.int16).tobytes()


async def broadcast_event(event_type: str, data: dict = None):
    """すべてのクライアントにイベントを送信"""
    payload = {"type": event_type}
    if data:
        payload.update(data)
    
    disconnected = set()
    for ws in active_connections:
        try:
            await ws.send_json(payload)
        except Exception:
            disconnected.add(ws)
    
    # 切断されたクライアントを削除
    active_connections.difference_update(disconnected)


async def broadcast_audio(audio: bytes):
    """すべてのクライアントに音声データを送信"""
    disconnected = set()
    for ws in active_connections:
        try:
            await ws.send_bytes(audio)
        except Exception:
            disconnected.add(ws)
    
    # 切断されたクライアントを削除
    active_connections.difference_update(disconnected)


class TextToSpeechSession:
    """WebSocket session for text-to-speech with ordered async queue"""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.llm: Optional[LLMService] = None
        self.tts: Optional[TTSService] = None
        self.t_start_ms: Optional[float] = None
        self.text_queue: asyncio.Queue = asyncio.Queue()  # テキスト入力キュー
        self.queue_processor_task: Optional[asyncio.Task] = None  # キュー処理タスク
        self.initialized = False

    async def start(self):
        """セッション開始"""
        try:
            print("[Session] Initializing LLM...")
            self.llm = LLMService()

            print("[Session] Initializing TTS...")
            self.tts = TTSService()

            self.t_start_ms = time.perf_counter() * 1000
            self.initialized = True
            
            # キュー処理タスクを開始
            self.queue_processor_task = asyncio.create_task(self._process_queue())
            
            print("[Session] All services initialized")
            await self.send_event("session_start", {"status": "ready", "sample_rate": 44100})
        except Exception as e:
            print(f"[Session] Init error: {e}")
            import traceback
            traceback.print_exc()
            await self.send_event("error", {"message": f"Init failed: {e}"})

    async def stop(self):
        """セッション終了"""
        # キュー処理タスクをキャンセル
        if self.queue_processor_task and not self.queue_processor_task.done():
            self.queue_processor_task.cancel()
        
        # LLMをキャンセル
        if self.llm:
            self.llm.request_cancel()

    async def send_event(self, name: str, payload: dict = None):
        """イベントをすべてのクライアントにブロードキャスト"""
        t = (time.perf_counter() * 1000 - self.t_start_ms) if self.t_start_ms else 0
        msg = {
            "type": "event",
            "name": name,
            "t": round(t),
            **(payload or {})
        }
        await broadcast_event("event", msg)

    async def send_audio(self, audio: np.ndarray):
        """音声をすべてのクライアントにブロードキャスト"""
        await broadcast_audio(np_int16_to_bytes(audio))

    async def handle_text_message(self, text: str):
        """テキストメッセージをキューに追加"""
        if not self.initialized:
            return

        if not text or not text.strip():
            return

        print(f"[Queue] Adding text: '{text}'")
        await self.send_event("text_queued", {"text": text, "queue_size": self.text_queue.qsize() + 1})
        
        # テキストをキューに追加（非同期）
        await self.text_queue.put(text)

    async def _process_queue(self):
        """キューからテキストを取り出して順に処理"""
        print("[Queue] Processor started")
        try:
            while True:
                # キューが空になるまで待機
                text = await self.text_queue.get()
                print(f"[Queue] Processing: '{text}' (remaining: {self.text_queue.qsize()})")
                
                try:
                    await self.run_llm_tts(text)
                except Exception as e:
                    print(f"[Queue] Processing error: {e}")
                    await self.send_event("error", {"message": f"Processing failed: {e}"})
                finally:
                    self.text_queue.task_done()
                    
        except asyncio.CancelledError:
            print("[Queue] Processor cancelled")
        except Exception as e:
            print(f"[Queue] Processor error: {e}")
            import traceback
            traceback.print_exc()

    async def run_llm_tts(self, user_text: str):
        """LLMストリーミング + 文単位TTS → WebSocket送信"""
        print(f"[LLM+TTS] Starting with: '{user_text}'")
        await self.send_event("llm_start", {"user_text": user_text})

        segmenter = SentenceSegmenter()
        was_cancelled = False

        try:
            async for delta in self.llm.generate_stream_async(user_text):
                for sentence in segmenter.push(delta):
                    await self.synthesize_and_send(sentence)

            for sentence in segmenter.flush_last():
                await self.synthesize_and_send(sentence)

        except asyncio.CancelledError:
            was_cancelled = True
            print(f"[LLM+TTS] Cancelled")
        except Exception as e:
            print(f"[LLM+TTS] Error: {e}")
            import traceback
            traceback.print_exc()
            await self.send_event("error", {"message": str(e)})
        finally:
            print(f"[LLM+TTS] Cleanup")
            if was_cancelled:
                await self.send_event("llm_end", {"reason": "cancelled"})
            else:
                await self.send_event("llm_end")

    async def synthesize_and_send(self, sentence: str):
        """1文をTTS合成してWebSocket送信"""
        print(f"[TTS] Synthesizing: '{sentence}'")
        await self.send_event("tts_start", {"text": sentence})

        loop = asyncio.get_event_loop()
        sr, audio = await loop.run_in_executor(
            None, lambda: self.tts.synthesize(sentence)
        )

        duration_ms = len(audio) * 1000 // sr
        print(f"[TTS] Done: {len(audio)} samples @ {sr}Hz, {len(audio)/sr:.2f}s")
        await self.send_event("tts_done", {
            "text": sentence,
            "duration_ms": duration_ms,
            "sample_rate": sr
        })

        await self.send_audio(audio)
        print(f"[TTS] Sent audio: {len(audio)} samples")


@app.websocket("/ws/tts")
async def websocket_text_to_speech(ws: WebSocket):
    """Text-to-Speech WebSocket endpoint."""
    await ws.accept()
    active_connections.add(ws)
    print(f"[WebSocket] Client connected. Total connections: {len(active_connections)}")

    session = TextToSpeechSession(ws)
    await session.start()

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if msg["type"] == "websocket.receive":
                data = msg.get("text") or msg.get("bytes")

                if isinstance(data, str):
                    try:
                        parsed = json.loads(data)
                        msg_type = parsed.get("type")
                        
                        if msg_type == "config":
                            await broadcast_event("config_ack", parsed)
                        elif msg_type == "ping":
                            await broadcast_event("pong", {})
                        elif msg_type == "text":
                            # テキストメッセージをキューに追加（ブロードキャスト）
                            text = parsed.get("text", "")
                            await session.handle_text_message(text)
                        elif msg_type == "cancel":
                            # キューを空にしてLLMをキャンセル
                            # （処理中のテキストはキャンセルできないが、キューの次のテキストは削除）
                            print(f"[Cancel] Clearing queue with {session.text_queue.qsize()} items")
                            while not session.text_queue.empty():
                                try:
                                    session.text_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                            if session.llm:
                                session.llm.request_cancel()
                    except json.JSONDecodeError:
                        pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        active_connections.discard(ws)
        print(f"[WebSocket] Client disconnected. Total connections: {len(active_connections)}")
        await session.stop()


@app.get("/")
async def root():
    """Root page."""
    return FileResponse(STATIC_DIR / "text_to_speech.html")


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "mode": "text-to-speech"}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
