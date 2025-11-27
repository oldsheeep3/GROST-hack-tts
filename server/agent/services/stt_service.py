"""Deepgram STT Service

Deepgram WebSocket API を用いたリアルタイムストリーミング STT。
PipecatSession から呼び出され、VAD が管理するユーザターン単位で
音声を送信し partial/final テキストを通知する。
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Protocol, runtime_checkable

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

from server.agent.vad_tracker import SAMPLE_RATE, FRAME_HOP_MS
from server.config import DEEPGRAM_API_KEY, DEEPGRAM_MODEL, DEEPGRAM_LANGUAGE


# =============================================================================
# Deepgram WebSocket URL
# =============================================================================
DEEPGRAM_WS_URL = "wss://api.deepgram.com/v1/listen"


def build_deepgram_url(
    model: str = DEEPGRAM_MODEL,
    language: str = DEEPGRAM_LANGUAGE,
    sample_rate: int = SAMPLE_RATE,
    encoding: str = "linear16",
    channels: int = 1,
    interim_results: bool = True,
    punctuate: bool = True,
    smart_format: bool = True,
    endpointing: int = 800,  # 800msの無音で speech_final（TURN_END_SOFT_MS=700に合わせる）
    vad_events: bool = True,
) -> str:
    """Deepgram WebSocket URLを構築"""
    params = [
        f"model={model}",
        f"language={language}",
        f"sample_rate={sample_rate}",
        f"encoding={encoding}",
        f"channels={channels}",
        f"interim_results={str(interim_results).lower()}",
        f"punctuate={str(punctuate).lower()}",
        f"smart_format={str(smart_format).lower()}",
        f"endpointing={endpointing}",
        f"vad_events={str(vad_events).lower()}",
    ]
    return f"{DEEPGRAM_WS_URL}?{'&'.join(params)}"


@runtime_checkable
class BaseSTTService(Protocol):
    """STT サービスの共通インターフェース"""

    on_speech_started: Optional[Callable[[], None]]
    on_speech_final: Optional[Callable[[str], None]]

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    def start_utterance(self) -> None:
        ...

    async def push_frame(self, pcm_frame, t_ms: int) -> None:
        ...

    async def get_all_partials(self) -> List[dict]:
        ...

    async def end_utterance(self, timeout_ms: int = 800) -> str:
        ...


@dataclass
class DeepgramSTTService:
    """Deepgram WebSocketストリーミングSTTクライアント

    Usage:
        stt = DeepgramSTTService()
        await stt.start()

        # USER_START 時
        stt.start_utterance()

        # フレーム受信時（VADがアクティブな時のみ）
        await stt.push_frame(pcm, t_ms)
        partials = await stt.get_all_partials()

        # USER_END_HARD 時
        final_text = await stt.end_utterance()

        # セッション終了時
        await stt.stop()
    """

    # 内部状態
    utterance_active: bool = False
    current_text: str = ""
    _accumulated_finals: List[str] = field(default_factory=list)

    # WebSocket接続
    _ws: Optional[Any] = field(default=None, repr=False)
    _receive_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _connected: bool = False

    # 非同期キュー（partial結果を蓄積）
    _partial_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # タイミング
    _utterance_start_time: Optional[float] = None
    _session_start_time: Optional[float] = None

    # 統計
    frames_sent: int = 0
    total_audio_ms: int = 0

    # コールバック（イベントを外部に通知）
    on_speech_started: Optional[Callable[[], None]] = field(default=None, repr=False)
    on_speech_final: Optional[Callable[[str], None]] = field(default=None, repr=False)  # speech_final時にテキストを通知

    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self):
        if not self._initialized:
            if not DEEPGRAM_API_KEY:
                raise ValueError("DEEPGRAM_API_KEY not set")
            self._initialized = True

    async def start(self):
        """セッション開始 - WebSocket接続を確立"""
        self._session_start_time = time.perf_counter()
        await self._connect()
        print("[STT] Session started (Deepgram WebSocket streaming)")

    async def stop(self):
        """セッション終了 - WebSocket接続を切断"""
        if self.utterance_active:
            await self.end_utterance()
        await self._disconnect()
        print("[STT] Session stopped")

    async def _connect(self):
        """Deepgram WebSocketに接続"""
        if self._connected:
            return

        url = build_deepgram_url()
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

        try:
            self._ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
            )
            self._connected = True
            print(f"[STT] Connected to Deepgram: {url}")

            # 受信タスクを開始
            self._receive_task = asyncio.create_task(self._receive_loop())

        except Exception as e:
            print(f"[STT] Connection failed: {e}")
            self._connected = False
            raise

    async def _disconnect(self):
        """WebSocket接続を切断"""
        self._connected = False

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        self._receive_task = None

        if self._ws:
            try:
                # CloseStreamメッセージを送信
                await self._ws.send(json.dumps({"type": "CloseStream"}))
                await self._ws.close()
            except Exception:
                pass
        self._ws = None

    async def _receive_loop(self):
        """WebSocketからの受信ループ"""
        try:
            async for message in self._ws:
                await self._handle_message(message)
        except ConnectionClosed as e:
            print(f"[STT] WebSocket closed: {e}")
            self._connected = False
            # 自動再接続を試みる
            await self._try_reconnect()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[STT] Receive error: {e}")
            import traceback
            traceback.print_exc()
            self._connected = False

    async def _try_reconnect(self, max_retries: int = 3, delay: float = 1.0):
        """接続断時の自動再接続"""
        for i in range(max_retries):
            try:
                print(f"[STT] Reconnecting... (attempt {i+1}/{max_retries})")
                await asyncio.sleep(delay)
                await self._connect()
                print(f"[STT] Reconnected successfully")
                return
            except Exception as e:
                print(f"[STT] Reconnect failed: {e}")
        print(f"[STT] Failed to reconnect after {max_retries} attempts")

    async def send_keepalive(self):
        """KeepAliveを送信（音声を送らない期間に呼ぶ）"""
        if self._ws and self._connected:
            try:
                await self._ws.send(json.dumps({"type": "KeepAlive"}))
            except Exception:
                pass

    async def _handle_message(self, message: str):
        """Deepgramからのメッセージを処理"""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            print(f"[STT] Failed to parse message: {message[:100]}")
            return

        msg_type = data.get("type", "")
        # デバッグ: 全メッセージタイプをログ
        print(f"[STT] Message type: {msg_type}")

        if msg_type == "Results":
            await self._handle_results(data)
        elif msg_type == "Metadata":
            # メタデータ（接続情報など）
            request_id = data.get("request_id", "")
            print(f"[STT] Metadata: request_id={request_id}")
        elif msg_type == "SpeechStarted":
            # DeepgramのVADが発話開始を検出 → コールバックで外部に通知
            print(f"[STT] SpeechStarted (utterance_active={self.utterance_active})")
            if self.on_speech_started and not self.utterance_active:
                # まだutteranceが開始されていない場合、外部に通知
                self.on_speech_started()
        elif msg_type == "UtteranceEnd":
            # 発話終了（endpointing）
            print(f"[STT] UtteranceEnd")
        elif msg_type == "Error":
            error = data.get("message", "Unknown error")
            print(f"[STT] Error from Deepgram: {error}")

    async def _handle_results(self, data: dict):
        """Results メッセージを処理

        ★ utterance_active チェックを廃止
        DeepgramのVADに任せて、常に結果を処理する。
        VADとSTTを並列で動かす設計。
        """
        channel = data.get("channel", {})
        alternatives = channel.get("alternatives", [])
        transcript = alternatives[0].get("transcript", "") if alternatives else ""

        if not alternatives:
            return

        confidence = alternatives[0].get("confidence", 0.0)
        is_final = data.get("is_final", False)
        speech_final = data.get("speech_final", False)

        # 空のトランスクリプトは無視
        if not transcript.strip():
            return

        # ログ出力
        flags = []
        if is_final:
            flags.append("is_final")
        if speech_final:
            flags.append("speech_final")
        flag_str = f" [{', '.join(flags)}]" if flags else " [interim]"
        print(f"[STT] '{transcript}'{flag_str}")

        # テキスト更新
        if is_final:
            # final結果を蓄積
            self._accumulated_finals.append(transcript)
            # 全体テキストを結合
            self.current_text = "".join(self._accumulated_finals)
        else:
            # interim結果は accumulated_finals + 今回のtranscriptで全体を構成
            self.current_text = "".join(self._accumulated_finals) + transcript

        # 差分計算（キューに入れる用）
        # 注: Deepgramは各結果で全体のtranscriptを返すので、差分計算は行わずそのまま
        await self._partial_queue.put({
            "delta": "",  # 差分は後で計算可能だが、今はtext全体を使う
            "text": self.current_text,
            "is_final": is_final,
            "speech_final": speech_final,
        })

        # ★ speech_final でコールバックを呼ぶ（LLM起動トリガー）
        if speech_final and self.on_speech_final and self.current_text.strip():
            print(f"[STT] speech_final → calling on_speech_final callback with text='{self.current_text}'")
            self.on_speech_final(self.current_text)

        # ★ speech_final でのリセットは行わない
        #    end_utterance() が呼ばれるまで current_text を保持する必要がある
        #    リセットは start_utterance() で行う

    def _reset_utterance(self):
        """発話状態をリセット"""
        self.utterance_active = False
        self.current_text = ""
        self._accumulated_finals.clear()
        self.frames_sent = 0
        self.total_audio_ms = 0
        self._utterance_start_time = None

        # キューをクリア
        while not self._partial_queue.empty():
            try:
                self._partial_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def start_utterance(self):
        """ターン開始 - USER_START 時に呼ぶ"""
        self._reset_utterance()
        self.utterance_active = True
        self._utterance_start_time = time.perf_counter()
        print(f"[STT] >>> start_utterance")

    async def end_utterance(self, timeout_ms: int = 1500) -> str:
        """ターン終了 - USER_END_HARD 時に呼ぶ

        最終的なSTT結果を返す。
        """
        if not self.utterance_active:
            return self.current_text

        # ★ utterance_active は最後まで True のままにする
        #   (Deepgramからの結果を受け取るため)

        # Deepgramに区切りを通知（FinalizeParagraph）
        if self._ws and self._connected:
            try:
                await self._ws.send(json.dumps({"type": "Finalize"}))
            except Exception as e:
                print(f"[STT] Failed to send Finalize: {e}")

        # 少し待って最終結果を受け取る
        try:
            await asyncio.wait_for(
                self._wait_for_final(),
                timeout=timeout_ms / 1000.0
            )
        except asyncio.TimeoutError:
            print(f"[STT] end_utterance: timeout waiting for final")

        # ★ 結果を受け取った後に utterance_active = False にする
        self.utterance_active = False

        elapsed = (time.perf_counter() - self._utterance_start_time) * 1000 if self._utterance_start_time else 0
        print(f"[STT] <<< end_utterance: text='{self.current_text}' (took {elapsed:.0f}ms)")

        return self.current_text

    async def _wait_for_final(self):
        """最終結果を待つ（speech_final=Trueまで）"""
        # 既にキューにある結果をチェック
        while True:
            try:
                p = self._partial_queue.get_nowait()
                if p.get("speech_final"):
                    return
            except asyncio.QueueEmpty:
                break

        # 新しい結果を待つ
        while self.utterance_active or not self._partial_queue.empty():
            try:
                p = await asyncio.wait_for(self._partial_queue.get(), timeout=0.1)
                if p.get("speech_final") or p.get("is_final"):
                    return
            except asyncio.TimeoutError:
                continue

    async def push_frame(self, pcm_frame: np.ndarray, t_now_ms: int):
        """20msフレームをDeepgramに送信"""
        # 接続がない場合は再接続を試みる
        if not self._connected or not self._ws:
            print(f"[STT] push_frame: not connected, trying to reconnect...")
            try:
                await self._connect()
            except Exception as e:
                print(f"[STT] push_frame: reconnect failed: {e}")
                return

        # int16 に変換
        if pcm_frame.dtype != np.int16:
            pcm_frame = pcm_frame.astype(np.int16)

        # バイト列に変換して送信
        pcm_bytes = pcm_frame.tobytes()

        try:
            await self._ws.send(pcm_bytes)
            self.frames_sent += 1
            self.total_audio_ms += FRAME_HOP_MS
        except Exception as e:
            print(f"[STT] Failed to send frame: {e}")
            self._connected = False

    async def get_partial(self) -> Optional[dict]:
        """partialを取得（非ブロッキング）"""
        try:
            return self._partial_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def get_all_partials(self) -> List[dict]:
        """溜まっているpartialを全て取得"""
        partials = []
        while True:
            p = await self.get_partial()
            if p is None:
                break
            partials.append(p)
        return partials

    def get_text(self) -> str:
        """現時点でのテキストを取得"""
        return self.current_text

    def get_stats(self) -> dict:
        """統計情報を取得"""
        return {
            "frames_sent": self.frames_sent,
            "total_audio_ms": self.total_audio_ms,
            "current_text": self.current_text,
            "utterance_active": self.utterance_active,
            "connected": self._connected,
        }


async def test_deepgram_stt_service():
    """DeepgramSTTService の簡易テスト"""
    import scipy.io.wavfile as wavfile
    import scipy.signal
    from server.config import OUTPUT_DIR

    print("=" * 60)
    print("Streaming STT Client Test (Deepgram WebSocket)")
    print("=" * 60)

    # 音声ファイルを読み込み
    input_path = OUTPUT_DIR / "test_input.wav"
    if not input_path.exists():
        print(f"Test file not found: {input_path}")
        return

    sr, audio = wavfile.read(input_path)
    print(f"Loaded: {sr}Hz, {len(audio)} samples, {len(audio)/sr:.2f}s")

    # リサンプリング
    if sr != SAMPLE_RATE:
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        num_samples = int(len(audio) * SAMPLE_RATE / sr)
        audio = scipy.signal.resample(audio, num_samples)
        audio = (audio * 32767).astype(np.int16)
        print(f"Resampled: {SAMPLE_RATE}Hz, {len(audio)} samples")

    # フレームに分割
    frame_samples = int(SAMPLE_RATE * FRAME_HOP_MS / 1000)
    frames = []
    for i in range(0, len(audio), frame_samples):
        frame = audio[i:i + frame_samples]
        if len(frame) < frame_samples:
            frame = np.pad(frame, (0, frame_samples - len(frame)), mode='constant')
        frames.append(frame)

    print(f"Split into {len(frames)} frames ({FRAME_HOP_MS}ms each)")
    print("-" * 60)

    # STTクライアント作成
    client = DeepgramSTTService()
    await client.start()

    try:
        # ターン開始
        client.start_utterance()

        # フレームを送信
        for i, frame in enumerate(frames):
            t_now_ms = i * FRAME_HOP_MS
            await client.push_frame(frame, t_now_ms)

            # partialをチェック
            partials = await client.get_all_partials()
            for p in partials:
                print(f"[{t_now_ms:5d}ms] STT: '{p['text']}' (final={p['is_final']})")

            # リアルタイムシミュレーション
            await asyncio.sleep(FRAME_HOP_MS / 1000)

        # ターン終了
        final_text = await client.end_utterance()

        print("-" * 60)
        print(f"Final text: {final_text}")
        print(f"Stats: {client.get_stats()}")

    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(test_deepgram_stt_service())
