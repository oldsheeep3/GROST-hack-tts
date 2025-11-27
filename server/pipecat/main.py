"""Pipecat-based WebSocket server for real-time voice agent.

★ 設計原則:
- 状態管理はTurnManagerに一元化
- main.pyはイベントのルーティングのみを担当
- Deepgramコールバックは即座にTurnManagerに委譲
"""
import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from server.agent.vad_tracker import VADFeatureTracker, SAMPLE_RATE, FRAME_HOP_MS
from server.agent.turn_manager import TurnManager, EventType, ConvState
from server.agent.segmenter import SentenceSegmenter
from server.agent.services.llm_service import LLMService
from server.agent.services.tts_service import TTSService
from server.agent.services.backchannel_service import BackchannelService
from server.agent.services.stt_service import BaseSTTService, DeepgramSTTService


app = FastAPI(title="Pipecat Voice Agent")

STATIC_DIR = Path(__file__).parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


def bytes_to_np_int16(pcm_bytes: bytes) -> np.ndarray:
    """bytes を int16 numpy array に変換"""
    return np.frombuffer(pcm_bytes, dtype=np.int16)


def np_int16_to_bytes(audio: np.ndarray) -> bytes:
    """int16 numpy array を bytes に変換"""
    return audio.astype(np.int16).tobytes()


class PipecatSession:
    """WebSocket session - 状態管理はTurnManagerに委譲"""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.vad: Optional[VADFeatureTracker] = None
        self.tm: Optional[TurnManager] = None
        self.stt: Optional[BaseSTTService] = None
        self.llm: Optional[LLMService] = None
        self.tts: Optional[TTSService] = None

        self.t_start_ms: Optional[float] = None
        self.frame_count = 0
        self.llm_task: Optional[asyncio.Task] = None
        self.initialized = False

        # 相槌サービス
        self.backchannel: Optional[BackchannelService] = None

        # ★ Deepgramコールバックからのイベントキュー（sync→async変換用）
        self._pending_events: list = []

    async def start(self):
        """セッション開始"""
        try:
            print("[Session] Initializing VAD...")
            self.vad = VADFeatureTracker()

            print("[Session] Initializing TurnManager...")
            self.tm = TurnManager()
            self.tm.tracker = self.vad

            print("[Session] Initializing STT...")
            self.stt = DeepgramSTTService()
            # ★ DeepgramコールバックをTurnManagerに委譲
            self.stt.on_speech_started = self._on_deepgram_speech_started
            self.stt.on_speech_final = self._on_deepgram_speech_final
            await self.stt.start()

            print("[Session] Initializing LLM...")
            self.llm = LLMService()

            print("[Session] Initializing TTS...")
            self.tts = TTSService()

            print("[Session] Initializing Backchannel...")
            self.backchannel = BackchannelService()
            self.backchannel.warmup(self.tts)

            self.t_start_ms = time.perf_counter() * 1000
            self.initialized = True
            print("[Session] All services initialized")
            await self.send_event("session_start", {"sample_rate": SAMPLE_RATE})
        except Exception as e:
            print(f"[Session] Init error: {e}")
            import traceback
            traceback.print_exc()
            await self.send_event("error", {"message": f"Init failed: {e}"})

    async def stop(self):
        """セッション終了"""
        if self.stt:
            await self.stt.stop()
        if self.llm_task and not self.llm_task.done():
            if self.llm:
                self.llm.request_cancel()
            self.llm_task.cancel()

    def _on_deepgram_speech_started(self):
        """DeepgramのSpeechStartedイベント（syncコールバック）

        ★ TurnManagerに委譲してイベントキューに追加
        """
        t_now_ms = self.frame_count * FRAME_HOP_MS
        events = self.tm.notify_deepgram_speech_started(t_now_ms)
        self._pending_events.extend(events)

        # STTのutteranceを開始
        if not self.stt.utterance_active:
            self.stt.start_utterance()

    def _on_deepgram_speech_final(self, text: str):
        """Deepgramのspeech_finalイベント（syncコールバック）

        ★ TurnManagerに委譲してイベントキューに追加
        """
        t_now_ms = self.frame_count * FRAME_HOP_MS
        events = self.tm.notify_deepgram_speech_final(text, t_now_ms)
        self._pending_events.extend(events)

    async def send_event(self, name: str, payload: dict = None):
        """イベントをクライアントに送信"""
        t = (time.perf_counter() * 1000 - self.t_start_ms) if self.t_start_ms else 0
        msg = {
            "type": "event",
            "name": name,
            "t": round(t),
            **(payload or {})
        }
        try:
            await self.ws.send_text(json.dumps(msg))
        except Exception:
            pass

    async def send_audio(self, audio: np.ndarray):
        """音声をクライアントに送信"""
        try:
            await self.ws.send_bytes(np_int16_to_bytes(audio))
        except Exception:
            pass

    async def handle_audio_frame(self, pcm_bytes: bytes):
        """20ms PCMフレームを処理

        ★ TurnManagerに状態管理を委譲したシンプルな実装
        """
        if not self.initialized:
            return

        pcm_np = bytes_to_np_int16(pcm_bytes)
        t_audio_ms = self.frame_count * FRAME_HOP_MS
        self.frame_count += 1
        current_time_ms = time.perf_counter() * 1000

        # 1. Deepgramコールバックからのイベントを処理
        await self._process_pending_events()

        # 2. VAD更新
        self.tm.tracker.update(pcm_np, t_audio_ms)

        # 3. エージェント発話中の割り込みチェック
        interrupt_events = self.tm.check_vad_interrupt(t_audio_ms)
        for ev_type, payload in interrupt_events:
            await self._handle_event(ev_type, payload)

        # 4. エージェント発話中でなければ通常のTurnManager処理
        if not self.tm.is_agent_speaking:
            events = self.tm.update_without_vad(t_audio_ms, "")
            for ev_type, payload in events:
                await self._handle_event(ev_type, payload)

        # 5. 常にDeepgramに音声を送信
        await self.stt.push_frame(pcm_np, t_audio_ms)

        # 6. STT partialを取得
        partials = await self.stt.get_all_partials()
        for p in partials:
            text = p.get("text", "")
            delta = p.get("delta", "")
            if delta:
                self.tm.stt_text += delta
            if text:
                await self.send_event("stt_partial", {"text": text})

        # 7. デバッグログ（VADまたはエージェント発話中）
        if self.tm.tracker.vad_user or self.tm.is_agent_speaking:
            print(f"[DEBUG] t={t_audio_ms}ms vad={self.tm.tracker.vad_user} state={self.tm.state.name} agent_speaking={self.tm.is_agent_speaking}")

    async def _process_pending_events(self):
        """Deepgramコールバックからのイベントを処理"""
        while self._pending_events:
            ev_type, payload = self._pending_events.pop(0)
            await self._handle_event(ev_type, payload)

    async def _handle_event(self, ev_type: EventType, payload: dict):
        """イベントを処理

        ★ TurnManagerからのイベントをハンドリング
        """
        await self.send_event(ev_type.name.lower(), payload)

        if ev_type == EventType.USER_START:
            # ユーザ発話開始 → STT utterance開始
            if not self.stt.utterance_active:
                self.stt.start_utterance()

        elif ev_type == EventType.AGENT_STOP_SPEAKING:
            # エージェント発話停止 → LLMタスクキャンセル
            if self.llm_task and not self.llm_task.done():
                if self.llm:
                    self.llm.request_cancel()
                self.llm_task.cancel()

        elif ev_type == EventType.AGENT_PAUSE:
            # 一時停止（クライアントに通知済み）
            pass

        elif ev_type == EventType.AGENT_RESUME:
            # 再開（クライアントに通知済み）
            pass

        elif ev_type == EventType.USER_END_HARD:
            # ユーザ発話終了確定 → LLM起動
            user_text = await self.stt.end_utterance(timeout_ms=800)
            if not user_text:
                user_text = self.tm.stt_text
            if user_text and user_text.strip():
                await self._start_llm_tts(user_text)

        elif ev_type == EventType.START_LLM:
            # Deepgram speech_finalからのLLM起動
            text = payload.get("text", "")
            if text and text.strip() and (self.llm_task is None or self.llm_task.done()):
                await self._start_llm_tts(text)

        elif ev_type == EventType.BC_WINDOW:
            # 相槌の窓
            if self.llm_task and not self.llm_task.done():
                return
            if self.tm.is_agent_speaking:
                return

            result = self.backchannel.get_random_audio()
            if result:
                phrase, sr, audio = result
                print(f"[Backchannel] Playing: '{phrase}'")
                audio_b64 = base64.b64encode(np_int16_to_bytes(audio)).decode('ascii')
                await self.send_event("backchannel", {
                    "text": phrase,
                    "audio": audio_b64,
                    "sample_rate": sr
                })

    async def _start_llm_tts(self, user_text: str):
        """LLM+TTSタスクを開始"""
        if self.llm_task and not self.llm_task.done():
            print(f"[LLM] Task already running, skipping")
            return

        self.tm.mark_llm_started()
        t_audio_ms = self.frame_count * FRAME_HOP_MS
        print(f"[LLM] Starting with: '{user_text}'")
        await self.send_event("llm_start", {"user_text": user_text})
        self.llm_task = asyncio.create_task(
            self.run_llm_tts(user_text, t_audio_ms)
        )

    def _should_continue_speaking(self) -> bool:
        """TTS を続けるべきか判定"""
        if self.llm_task is None:
            return True
        return not self.llm_task.cancelled()

    async def run_llm_tts(self, user_text: str, t_audio_ms: float):
        """LLMストリーミング + 文単位TTS → WebSocket送信"""
        print(f"[LLM+TTS] Starting with: '{user_text}'")

        # ★ TurnManagerにエージェント発話開始を通知
        current_time_ms = time.perf_counter() * 1000
        self.tm.notify_agent_start_speaking(0, int(current_time_ms))

        segmenter = SentenceSegmenter()
        was_cancelled = False
        early_exit = False

        try:
            async for delta in self.llm.generate_stream_async(user_text):
                for sentence in segmenter.push(delta):
                    if not self._should_continue_speaking():
                        early_exit = True
                        return
                    await self.synthesize_and_send(sentence)

            for sentence in segmenter.flush_last():
                if not self._should_continue_speaking():
                    early_exit = True
                    return
                await self.synthesize_and_send(sentence)

        except asyncio.CancelledError:
            was_cancelled = True
            print(f"[LLM+TTS] Cancelled by user interruption")
        except Exception as e:
            await self.send_event("error", {"message": str(e)})
        finally:
            # ★ TurnManagerにエージェント発話完了を通知
            current_time_ms = time.perf_counter() * 1000
            self.tm.notify_agent_done_speaking(int(current_time_ms))
            print(f"[LLM+TTS] Cleanup: agent done speaking")

            if was_cancelled:
                pass
            elif early_exit:
                print(f"[LLM+TTS] Early exit, sending llm_end")
                await self.send_event("llm_end", {"reason": "early_exit"})
            else:
                self.tm.reset_for_new_turn()
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

        # ★ TurnManagerに再生時間を通知
        current_time_ms = time.perf_counter() * 1000
        self.tm.notify_agent_start_speaking(duration_ms, int(current_time_ms))


@app.websocket("/ws/realtime")
async def websocket_realtime(ws: WebSocket):
    """Real-time voice WebSocket endpoint."""
    await ws.accept()

    session = PipecatSession(ws)
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
                        if parsed.get("type") == "config":
                            await session.send_event("config_ack", parsed)
                        elif parsed.get("type") == "ping":
                            await session.send_event("pong")
                    except json.JSONDecodeError:
                        pass
                    continue

                if isinstance(data, bytes):
                    await session.handle_audio_frame(data)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await session.stop()


@app.get("/")
async def root():
    """Root page."""
    return FileResponse(STATIC_DIR / "realtime.html")


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "implementation": "pipecat"}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
