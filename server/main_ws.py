"""リアルタイム音声対話 WebSocket サーバ

通信仕様:
- クライアント → サーバ
  - text: {"type": "config", "sampleRate": 16000} など
  - binary: 16kHz mono PCM (Int16) 20ms フレーム

- サーバ → クライアント
  - text: {"type": "event", "name": "...", "t": 1234, ...}
  - binary: 16kHz mono PCM (Int16) TTS出力
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
from fastapi.responses import HTMLResponse, FileResponse

from server.agent.vad_tracker import VADFeatureTracker, SAMPLE_RATE, FRAME_HOP_MS
from server.agent.turn_manager import TurnManager, EventType, ConvState
from server.agent.stt_client import StreamingSTTClient
from server.agent.segmenter import SentenceSegmenter
from server.agent.services.llm_service import LLMService
from server.agent.services.tts_service import TTSService
from server.agent.services.backchannel_service import BackchannelService


app = FastAPI(title="Realtime Voice Agent")

# 静的ファイル
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


def bytes_to_np_int16(pcm_bytes: bytes) -> np.ndarray:
    """bytes を int16 numpy array に変換"""
    return np.frombuffer(pcm_bytes, dtype=np.int16)


def np_int16_to_bytes(audio: np.ndarray) -> bytes:
    """int16 numpy array を bytes に変換"""
    return audio.astype(np.int16).tobytes()


class RealtimeSession:
    """1つのWebSocket接続に紐づくセッション"""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.vad = None
        self.tm = None
        self.stt = None
        self.llm = None
        self.tts = None

        self.t_start_ms: Optional[float] = None
        self.frame_count = 0
        self.llm_task: Optional[asyncio.Task] = None
        self.is_agent_speaking = False
        self.initialized = False

        # 割り込み判定用
        self.interrupt_vad_start_ms: Optional[int] = None  # VAD開始時刻
        self.interrupt_text: str = ""  # 割り込み中のSTTテキスト
        self.is_paused: bool = False  # 一時停止中フラグ

        # 再生終了予想時刻（クライアント側の再生完了を推定）
        self.playback_end_time_ms: float = 0

        # 相槌サービス（シングルトン、キャッシュ済み音声を使用）
        self.backchannel: BackchannelService = None

    async def start(self):
        """セッション開始"""
        try:
            print("[Session] Initializing VAD...")
            self.vad = VADFeatureTracker()

            print("[Session] Initializing TurnManager...")
            self.tm = TurnManager()
            self.tm.tracker = self.vad

            print("[Session] Initializing STT...")
            self.stt = StreamingSTTClient()
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
                self.llm.request_cancel()  # LLMスレッドにキャンセルを要求
            self.llm_task.cancel()

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
        """20ms PCMフレームを処理"""
        if not self.initialized:
            return

        pcm_np = bytes_to_np_int16(pcm_bytes)

        # フレーム時刻
        t_audio_ms = self.frame_count * FRAME_HOP_MS
        self.frame_count += 1

        # 1. VAD判定は常に実行（エージェント発話中も監視）
        self.tm.tracker.update(pcm_np, t_audio_ms)
        vad_active = self.tm.tracker.vad_user

        # 現在時刻（ミリ秒）
        current_time_ms = time.perf_counter() * 1000

        # 2. エージェント発話中の処理
        #    ★ 設計思想に合わせた3段階判定:
        #       - 200ms以上: 一時停止（まだ相槌の可能性あり）
        #       - 500ms以上 or STT認識開始: 完全停止（本格的な割り込み）
        #       - VAD終了時に200ms未満: 相槌として無視（何もしない）
        #       - VAD終了時に200-500ms: 一時停止中なら再開
        is_playing = self.is_agent_speaking or current_time_ms < self.playback_end_time_ms
        if is_playing:
            # 音声は常にDeepgramに送信
            await self.stt.push_frame(pcm_np, t_audio_ms)

            if vad_active:
                # VAD開始時刻を記録
                if self.interrupt_vad_start_ms is None:
                    self.interrupt_vad_start_ms = t_audio_ms
                    print(f"[Interrupt] VAD started at {t_audio_ms}ms")

                # VAD継続時間
                vad_duration_ms = t_audio_ms - self.interrupt_vad_start_ms

                # ★ 200ms以上で一時停止（相槌の上限に達した）
                if vad_duration_ms >= 200 and not self.is_paused:
                    self.is_paused = True
                    await self.send_event("agent_paused", {"reason": "vad_detected"})
                    print(f"[Interrupt] Audio paused at {vad_duration_ms}ms (waiting for STT or timeout)")

                # ★ 500ms以上のVADで割り込み確定 → 完全停止
                #    （STTが認識を返し始める目安）
                if vad_duration_ms >= 500:
                    print(f"[Interrupt] Stopping agent: VAD duration={vad_duration_ms}ms (timeout)")
                    self.is_agent_speaking = False
                    self.playback_end_time_ms = 0
                    self.interrupt_vad_start_ms = None
                    self.is_paused = False
                    if self.llm_task and not self.llm_task.done():
                        if self.llm:
                            self.llm.request_cancel()  # LLMスレッドにキャンセルを要求
                        self.llm_task.cancel()
                        # キャンセル時は明示的にllm_endを送信（タスクのfinallyが実行される前にクライアントに通知）
                        await self.send_event("llm_end", {"reason": "interrupted"})
                    await self.send_event("agent_interrupted", {"reason": "user_speaking"})
                    # ここでreturnせず、下のTurnManager処理に流す
                else:
                    # まだ500ms未満 → 次フレームで再判定
                    return
            else:
                # VADが非アクティブになった
                if self.interrupt_vad_start_ms is not None:
                    vad_duration_ms = t_audio_ms - self.interrupt_vad_start_ms
                    self.interrupt_vad_start_ms = None

                    if vad_duration_ms < 200:
                        # ★ 200ms未満: 相槌として完全無視（一時停止もしてない）
                        print(f"[Interrupt] VAD ended (duration={vad_duration_ms}ms) → short backchannel, ignored")
                    elif self.is_paused:
                        # ★ 200-500msでVAD終了: 一時停止中なら再開
                        print(f"[Interrupt] VAD ended (duration={vad_duration_ms}ms) → backchannel, resuming")
                        self.is_paused = False
                        await self.send_event("agent_resumed", {"reason": "backchannel"})
                    else:
                        print(f"[Interrupt] VAD ended (duration={vad_duration_ms}ms)")

                # エージェント発話中はTurnManager処理をスキップ
                return

        # 4. TurnManager更新（VADは既に更新済みなのでスキップ）
        #    ★ push_frame より先に実行して、USER_START で start_utterance() を呼ぶ
        events = self.tm.update_without_vad(t_audio_ms, "")

        # デバッグ: VADとイベントの状態
        if vad_active or events:
            print(f"[DEBUG] t={t_audio_ms}ms vad={vad_active} events={[e[0].name for e in events]} state={self.tm.state.name}")

        # 5. イベント処理（STTより先に）
        for ev_type, payload in events:
            await self.send_event(ev_type.name.lower(), payload)

            if ev_type == EventType.USER_START:
                # ユーザが喋り始めた → 新しいターン開始
                print(f"[DEBUG] USER_START → calling stt.start_utterance()")
                self.stt.start_utterance()
                print(f"[DEBUG] stt.utterance_active = {self.stt.utterance_active}")

                if self.is_agent_speaking:
                    # エージェントが喋っている最中なら停止
                    self.is_agent_speaking = False
                    if self.llm_task and not self.llm_task.done():
                        if self.llm:
                            self.llm.request_cancel()  # LLMスレッドにキャンセルを要求
                        self.llm_task.cancel()
                    await self.send_event("agent_interrupted")

            elif ev_type == EventType.USER_END_HARD:
                # ★ STTが開始されていない場合は無視（妄想発火防止）
                if not self.stt.utterance_active:
                    print(f"[Turn] ignore USER_END_HARD: no active utterance")
                    continue

                # ユーザ発話終了確定 → STT を end して LLM+TTS開始
                user_text = await self.stt.end_utterance(timeout_ms=800)
                if not user_text:
                    user_text = self.tm.stt_text
                if self.llm_task is None or self.llm_task.done():
                    if user_text and user_text.strip():
                        self.llm_task = asyncio.create_task(
                            self.run_llm_tts(user_text, t_audio_ms)
                        )

            elif ev_type == EventType.BC_WINDOW:
                # エージェント側の相槌（ユーザー発話中に「うんうん」等を挟む）
                if self.llm_task and not self.llm_task.done():
                    continue
                if self.is_agent_speaking:
                    continue

                # キャッシュ済み音声を取得（クールダウン込み）
                result = self.backchannel.get_random_audio()
                if result:
                    phrase, sr, audio = result
                    print(f"[Backchannel] Playing: '{phrase}'")
                    # backchannelイベントに音声データを埋め込む（Base64）
                    # メインTTSキューとは別経路で即座に再生させる
                    audio_b64 = base64.b64encode(np_int16_to_bytes(audio)).decode('ascii')
                    await self.send_event("backchannel", {
                        "text": phrase,
                        "audio": audio_b64,
                        "sample_rate": sr
                    })

        # 6. 常にDeepgramに音声を送信（VADに関係なく）
        #    ★ Deepgramの内部VADを活用し、発話の頭・末尾の欠落を防止
        await self.stt.push_frame(pcm_np, t_audio_ms)

        # 7. STT partialを取得してTurnManagerに渡す
        stt_partial = ""
        partials = await self.stt.get_all_partials()
        for p in partials:
            stt_partial += p.get("delta", "")
            text = p.get("text", "")
            if text:
                print(f"[STT] partial: {text}")
                await self.send_event("stt_partial", {"text": text})

        # STT結果をTurnManagerにも反映
        if stt_partial:
            self.tm.stt_text += stt_partial

        # 8. LLM起動判定（USER_END_HARDを待たずに早期起動）
        if (self.llm_task is None or self.llm_task.done()) and self.tm.should_start_llm():
            if not self.tm.llm_started:
                self.tm.mark_llm_started()
                user_text = self.tm.stt_text or self.stt.get_text()
                print(f"[LLM] should_start_llm=True, user_text='{user_text}'")
                if user_text.strip():
                    print(f"[LLM] Starting LLM+TTS task")
                    await self.send_event("llm_start_early", {"user_text": user_text})
                    self.llm_task = asyncio.create_task(
                        self.run_llm_tts(user_text, t_audio_ms)
                    )

    def _should_continue_speaking(self) -> bool:
        """TTS を続けるべきか判定（キャンセルされたら False）"""
        if self.llm_task is None:
            return True
        return not self.llm_task.cancelled()

    def _should_interrupt(self, vad_duration_ms: int, stt_text: str) -> bool:
        """割り込みすべきかどうか判定

        判定基準:
        1. VAD継続時間が長い（800ms以上） → 割り込み意図が高い
        2. STTテキストが相槌パターンでない → 割り込み意図が高い
        3. STTテキストが長い（5文字以上） → 割り込み意図が高い
        """
        # 相槌パターン（これらは割り込みとみなさない）
        BACKCHANNEL_PATTERNS = [
            "うん", "ううん", "はい", "へー", "ふーん", "そう", "そうそう",
            "なるほど", "えー", "あー", "おー", "ほー", "わかる", "たしかに",
            "そうだね", "そうですね", "ですね", "ね", "うーん", "まあ",
        ]

        # 1. 長時間VAD → 確実に割り込み
        if vad_duration_ms >= 800:
            print(f"[Interrupt] Long VAD duration: {vad_duration_ms}ms")
            return True

        # 2. STTテキストがある場合の判定
        if stt_text:
            text_normalized = stt_text.strip().lower()

            # 相槌パターンに完全一致 → 割り込みではない
            for pattern in BACKCHANNEL_PATTERNS:
                if text_normalized == pattern or text_normalized.endswith(pattern):
                    # ただし500ms以上続いたら割り込み扱い
                    if vad_duration_ms >= 500:
                        print(f"[Interrupt] Backchannel but long: {vad_duration_ms}ms")
                        return True
                    return False

            # 5文字以上 → 割り込み意図が高い
            if len(text_normalized) >= 5:
                print(f"[Interrupt] Long text: '{stt_text}'")
                return True

            # 割り込みキーワード
            INTERRUPT_KEYWORDS = ["待って", "ちょっと", "あのさ", "すみません", "ごめん", "違う"]
            for keyword in INTERRUPT_KEYWORDS:
                if keyword in text_normalized:
                    print(f"[Interrupt] Interrupt keyword: '{keyword}'")
                    return True

        # 3. VADが400ms以上でSTTテキストがある → 割り込み可能性高い
        if vad_duration_ms >= 400 and stt_text:
            print(f"[Interrupt] Medium VAD with text: {vad_duration_ms}ms '{stt_text}'")
            return True

        # まだ判定保留
        return False

    async def run_llm_tts(self, user_text: str, t_audio_ms: float):
        """LLMストリーミング + 文単位TTS → WebSocket送信"""
        print(f"[LLM+TTS] Starting with: '{user_text}'")
        # ★ LLM開始時点で speaking フラグを立てる（割り込み検知を有効化）
        self.is_agent_speaking = True
        segmenter = SentenceSegmenter()
        was_cancelled = False
        early_exit = False  # _should_continue_speaking()がFalseでreturnした場合

        await self.send_event("llm_start", {"user_text": user_text})

        try:
            async for delta in self.llm.generate_stream_async(user_text):
                # 文単位でTTS
                for sentence in segmenter.push(delta):
                    if not self._should_continue_speaking():
                        early_exit = True
                        return
                    await self.synthesize_and_send(sentence)

            # 最後の文
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
            self.is_agent_speaking = False
            # ★ 常にplayback_end_time_msをリセット（ハング防止の最重要ポイント）
            self.playback_end_time_ms = 0
            print(f"[LLM+TTS] Cleanup: is_agent_speaking=False, playback_end_time_ms=0")

            # ★ キャンセル時はリセットしない（TurnManagerが既にLISTENINGに遷移している）
            #   また、キャンセル時はllm_endは既に割り込み処理で送信済み
            if was_cancelled:
                pass  # 割り込みで既にllm_end送信済み
            elif early_exit:
                # 早期終了（_should_continue_speaking()=False）の場合もllm_endを送信
                print(f"[LLM+TTS] Early exit, sending llm_end")
                await self.send_event("llm_end", {"reason": "early_exit"})
            else:
                # 正常終了
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
            "sample_rate": sr  # サンプルレートも送る
        })

        # ★ 一括送信（チャンク分割をやめてブツブツの原因を排除）
        await self.send_audio(audio)
        print(f"[TTS] Sent audio: {len(audio)} samples")

        # ★ 再生終了予想時刻を更新（現在時刻 + 音声長さ + バッファマージン100ms）
        current_time_ms = time.perf_counter() * 1000
        new_end_time = current_time_ms + duration_ms + 100
        if new_end_time > self.playback_end_time_ms:
            self.playback_end_time_ms = new_end_time
            print(f"[TTS] Playback end estimated at {self.playback_end_time_ms:.0f}ms")


@app.websocket("/ws/realtime")
async def websocket_realtime(ws: WebSocket):
    """リアルタイム音声対話WebSocketエンドポイント"""
    await ws.accept()

    session = RealtimeSession(ws)
    await session.start()

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if msg["type"] == "websocket.receive":
                data = msg.get("text") or msg.get("bytes")

                # text: 設定/制御メッセージ
                if isinstance(data, str):
                    try:
                        parsed = json.loads(data)
                        if parsed.get("type") == "config":
                            # 設定メッセージ（今は何もしない）
                            await session.send_event("config_ack", parsed)
                        elif parsed.get("type") == "ping":
                            await session.send_event("pong")
                    except json.JSONDecodeError:
                        pass
                    continue

                # binary: PCMフレーム
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
    """ルートページ → realtime.htmlにリダイレクト"""
    return FileResponse(STATIC_DIR / "realtime.html")


@app.get("/health")
async def health():
    """ヘルスチェック"""
    return {"status": "ok"}


# 静的ファイルをマウント
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
