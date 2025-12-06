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
    """WebSocket session for text-to-speech with single LLM and multiple TTS workers"""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.llm: Optional[LLMService] = None
        self.tts: Optional[TTSService] = None
        self.t_start_ms: Optional[float] = None
        
        # 入力キュー（ユーザーのテキスト入力）
        self.text_input_queue: asyncio.Queue = asyncio.Queue()
        
        # 中間キュー（LLMが生成した文章）
        self.generated_text_queue: asyncio.Queue = asyncio.Queue()
        
        # ワーカータスク
        self.llm_task: Optional[asyncio.Task] = None  # LLM は単一
        self.tts_workers: list[asyncio.Task] = []     # TTS は複数
        
        self.initialized = False
        self.num_tts_workers = 3   # TTS 並列度（音声生成を複数並列）

    async def start(self):
        """セッション開始"""
        try:
            print("[Session] Initializing LLM...")
            self.llm = LLMService()

            print("[Session] Initializing TTS...")
            self.tts = TTSService()

            self.t_start_ms = time.perf_counter() * 1000
            self.initialized = True
            
            # LLM は単一で実行（入力キューから順番に処理）
            self.llm_task = asyncio.create_task(self._llm_processor())
            
            # TTS ワーカーを複数起動（並列で音声生成）
            for i in range(self.num_tts_workers):
                worker_task = asyncio.create_task(self._tts_worker(i))
                self.tts_workers.append(worker_task)
            
            print(f"[Session] All services initialized: 1 LLM + {self.num_tts_workers} TTS workers")
            await self.send_event("session_start", {
                "status": "ready",
                "sample_rate": 44100,
                "tts_workers": self.num_tts_workers
            })
        except Exception as e:
            print(f"[Session] Init error: {e}")
            import traceback
            traceback.print_exc()
            await self.send_event("error", {"message": f"Init failed: {e}"})

    async def stop(self):
        """セッション終了"""
        # LLM タスクをキャンセル
        if self.llm_task and not self.llm_task.done():
            self.llm_task.cancel()
        
        # すべての TTS ワーカータスクをキャンセル
        for worker_task in self.tts_workers:
            if worker_task and not worker_task.done():
                worker_task.cancel()
        
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
        """テキストメッセージを入力キューに追加"""
        if not self.initialized:
            return

        if not text or not text.strip():
            return

        print(f"[Input] Adding text: '{text[:50]}...'")
        queue_size = self.text_input_queue.qsize() + 1
        await self.send_event("text_queued", {"text": text, "queue_size": queue_size})
        
        # テキストを入力キューに追加（非同期）
        await self.text_input_queue.put(text)

    async def _llm_processor(self):
        """LLM プロセッサ：テキストを順番に処理して文を生成
        
        LLM は単一で動作し、入力キューから順番にテキストを処理。
        生成した文を即座に中間キューに追加（TTS ワーカーが並列処理）
        """
        print("[LLM] Processor started (single threaded)")
        try:
            while True:
                # 入力キューからテキストを取得（FIFO）
                user_text = await self.text_input_queue.get()
                try:
                    print(f"[LLM] Processing text: '{user_text[:50]}...'")
                    await self.send_event("llm_start", {"user_text": user_text})
                    
                    # LLM で文章を生成（ストリーミング）
                    segmenter = SentenceSegmenter()
                    sentence_count = 0
                    
                    async for delta in self.llm.generate_stream_async(user_text):
                        for sentence in segmenter.push(delta):
                            sentence_count += 1
                            # 生成された文を即座に TTS キューに追加
                            await self.generated_text_queue.put(sentence)
                            print(f"[LLM] Sentence {sentence_count}: '{sentence[:50]}...'")
                    
                    # 最後の文を取得
                    for sentence in segmenter.flush_last():
                        sentence_count += 1
                        await self.generated_text_queue.put(sentence)
                        print(f"[LLM] Sentence {sentence_count}: '{sentence[:50]}...'")
                    
                    print(f"[LLM] Completed: generated {sentence_count} sentences")
                    await self.send_event("llm_end", {"sentences": sentence_count})
                    
                except Exception as e:
                    print(f"[LLM] Error: {e}")
                    import traceback
                    traceback.print_exc()
                    await self.send_event("error", {"message": f"LLM error: {e}"})
                finally:
                    self.text_input_queue.task_done()
                    
        except asyncio.CancelledError:
            print("[LLM] Processor cancelled")
        except Exception as e:
            print(f"[LLM] Unexpected error: {e}")
            import traceback
            traceback.print_exc()

    async def _tts_worker(self, worker_id: int):
        """TTS ワーカー：文章を受け取って音声を生成・送信"""
        print(f"[TTS-Worker-{worker_id}] Started")
        try:
            while True:
                # 生成キューから文を取得
                sentence = await self.generated_text_queue.get()
                try:
                    print(f"[TTS-Worker-{worker_id}] Synthesizing: '{sentence[:50]}...'")
                    await self.send_event("tts_start", {"text": sentence})
                    
                    # TTS で音声を生成
                    loop = asyncio.get_event_loop()
                    sr, audio = await loop.run_in_executor(
                        None, lambda: self.tts.synthesize(sentence)
                    )
                    
                    duration_ms = len(audio) * 1000 // sr
                    print(f"[TTS-Worker-{worker_id}] Done: {len(audio)} samples @ {sr}Hz, {duration_ms}ms")
                    await self.send_event("tts_done", {
                        "text": sentence,
                        "duration_ms": duration_ms,
                        "sample_rate": sr
                    })
                    
                    # 音声をすべてのクライアントにブロードキャスト
                    await self.send_audio(audio)
                    
                except Exception as e:
                    print(f"[TTS-Worker-{worker_id}] Error: {e}")
                    await self.send_event("error", {"message": f"TTS Worker-{worker_id}: {e}"})
                finally:
                    self.generated_text_queue.task_done()
                    
        except asyncio.CancelledError:
            print(f"[TTS-Worker-{worker_id}] Cancelled")
        except Exception as e:
            print(f"[TTS-Worker-{worker_id}] Unexpected error: {e}")
            import traceback
            traceback.print_exc()


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
