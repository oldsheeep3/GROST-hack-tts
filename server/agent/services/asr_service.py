"""Qwen3-ASR Service using DashScope MultiModalConversation API"""
import asyncio
import os
import tempfile
import time
import wave
from typing import Iterator, AsyncIterator
from queue import Queue
from threading import Thread

import dashscope
from dashscope import MultiModalConversation

from server.config import DASHSCOPE_API_KEY, QWEN_ASR_MODEL


def write_pcm16le_to_wav(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    num_channels: int = 1,
) -> str:
    """16kHz mono PCM を一時 WAV ファイルに吐く"""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # 16bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return path


class ASRService:
    """Qwen3-ASR をラップしたASRサービス（DashScope SDK直叩き）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        if not DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY environment variable is not set")

        dashscope.api_key = DASHSCOPE_API_KEY
        dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
        self._initialized = True
        print("ASR service initialized (Qwen3-ASR).")

    def transcribe(self, audio_path: str, language: str = "ja", context: str = "") -> str:
        """音声ファイルからテキストに変換（同期版）"""
        messages = [
            {
                "role": "system",
                "content": [{"text": context}],
            },
            {
                "role": "user",
                "content": [{"audio": audio_path}],
            },
        ]

        resp = MultiModalConversation.call(
            model=QWEN_ASR_MODEL,
            messages=messages,
            result_format="message",
        )

        if getattr(resp, "status_code", 500) != 200:
            raise RuntimeError(f"Qwen ASR failed: {getattr(resp, 'message', 'unknown error')}")

        choice = resp.output.choices[0]
        content = choice.message.content[0]
        text = content.get("text", "")
        return text

    def transcribe_stream(self, audio_path: str, language: str = "ja", context: str = "") -> Iterator[str]:
        """音声ファイルからテキストに変換（ストリーミング版）

        DashScope SDK の stream=True + incremental_output=True で
        本物のストリーミングを実現。差分だけをyieldする。
        """
        messages = [
            {
                "role": "system",
                "content": [{"text": context}],
            },
            {
                "role": "user",
                "content": [{"audio": audio_path}],
            },
        ]

        # stream=True + incremental_output=True で本物のストリーミング
        resp_iter = MultiModalConversation.call(
            model=QWEN_ASR_MODEL,
            messages=messages,
            result_format="message",
            stream=True,
            incremental_output=True,  # これが重要：増分出力を有効化
        )

        last_text = ""
        for chunk in resp_iter:
            try:
                choice = chunk.output.choices[0]
                content = choice.message.content[0]
                text_now = content.get("text", "")

                if not text_now:
                    continue

                # 差分を計算：APIが全文を返す場合と差分を返す場合の両方に対応
                if text_now.startswith(last_text):
                    # 全文が返ってくるパターン：差分を抽出
                    delta = text_now[len(last_text):]
                else:
                    # 差分が直接返ってくるパターン
                    delta = text_now

                last_text = text_now if text_now.startswith(last_text) else last_text + text_now

                if delta:
                    yield delta
            except Exception as e:
                # エラーは無視して続行
                continue

    def transcribe_audio(self, audio_array, sample_rate: int = 16000, language: str = "ja") -> str:
        """音声配列からテキストに変換"""
        import numpy as np

        if isinstance(audio_array, np.ndarray):
            if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                audio_array = (audio_array * 32767).astype(np.int16)
            pcm_bytes = audio_array.tobytes()
        else:
            pcm_bytes = audio_array

        wav_path = write_pcm16le_to_wav(pcm_bytes, sample_rate)
        try:
            return self.transcribe(wav_path, language)
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    async def transcribe_stream_async(
        self, audio_path: str, language: str = "ja", context: str = ""
    ) -> AsyncIterator[str]:
        """音声ファイルからテキストに変換（非同期ストリーミング版）

        同期のストリーミングを別スレッドで実行し、Queueを経由して
        チャンクが到着するたびにyieldする（本物のストリーミング）。
        """
        queue: Queue = Queue()
        finished = {"done": False, "error": None}

        def run_sync_stream():
            try:
                for chunk in self.transcribe_stream(audio_path, language, context):
                    queue.put(("chunk", chunk))
            except Exception as e:
                finished["error"] = e
            finally:
                finished["done"] = True
                queue.put(("done", None))

        # 別スレッドでストリーミングを実行
        thread = Thread(target=run_sync_stream)
        thread.start()

        # チャンクが到着するたびにyield
        while True:
            # 短いポーリング間隔でキューをチェック
            await asyncio.sleep(0.01)

            while not queue.empty():
                msg_type, data = queue.get_nowait()
                if msg_type == "chunk":
                    yield data
                elif msg_type == "done":
                    thread.join()
                    if finished["error"]:
                        raise finished["error"]
                    return

    async def transcribe_pcm_stream_async(
        self, pcm_bytes: bytes, sample_rate: int = 16000, language: str = "ja"
    ) -> AsyncIterator[str]:
        """PCMバイト列からテキストに変換（非同期ストリーミング版）"""
        wav_path = write_pcm16le_to_wav(pcm_bytes, sample_rate)
        try:
            async for chunk in self.transcribe_stream_async(wav_path, language):
                yield chunk
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
