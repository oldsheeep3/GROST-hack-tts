"""Gemini LLM Service using google-genai SDK"""
import asyncio
from typing import Iterator, AsyncIterator, List, Dict, Any

from google import genai
from google.genai import types as genai_types

from server.config import GEMINI_API_KEY, GEMINI_MODEL, SYSTEM_PROMPT, LLM_MAX_TOKENS


def build_chat_contents(history: List[Dict[str, Any]], user_text: str) -> list:
    """
    history: [{"role": "user"|"model", "content": "..."}, ...]
    を Gemini API の contents 形式に変換
    """
    contents = []
    for msg in history:
        contents.append({
            "role": msg["role"],
            "parts": [{"text": msg["content"]}],
        })
    contents.append({
        "role": "user",
        "parts": [{"text": user_text}],
    })
    return contents


class LLMService:
    """Gemini APIをラップしたLLMサービス（google-genai SDK使用）"""

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = GEMINI_MODEL
        self.history: List[Dict[str, str]] = []
        self._cancel_requested = False  # キャンセルフラグ
        print("LLM service initialized.")

    def request_cancel(self):
        """ストリーミング中断をリクエスト"""
        self._cancel_requested = True

    def _reset_cancel(self):
        """キャンセルフラグをリセット"""
        self._cancel_requested = False

    def generate(self, user_input: str) -> str:
        """ユーザー入力に対する応答を生成（同期版）

        Args:
            user_input: ユーザーの発話テキスト

        Returns:
            生成された応答テキスト
        """
        contents = build_chat_contents(self.history, user_input)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=LLM_MAX_TOKENS,
            ),
        )

        response_text = response.text

        # 履歴に追加
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "model", "content": response_text})

        return response_text

    def generate_stream(self, user_input: str) -> Iterator[str]:
        """ユーザー入力に対する応答をストリーミング生成

        Args:
            user_input: ユーザーの発話テキスト

        Yields:
            生成されたテキストのチャンク
        """
        contents = build_chat_contents(self.history, user_input)

        stream = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=LLM_MAX_TOKENS,
            ),
        )

        full_response = []
        for chunk in stream:
            text_piece = getattr(chunk, "text", None)
            if text_piece:
                full_response.append(text_piece)
                yield text_piece

        # 履歴に追加
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "model", "content": "".join(full_response)})

    def reset(self):
        """会話履歴をリセット"""
        self.history = []

    async def generate_stream_async(self, user_input: str) -> AsyncIterator[str]:
        """ユーザー入力に対する応答を非同期ストリーミング生成

        Args:
            user_input: ユーザーの発話テキスト

        Yields:
            生成されたテキストのチャンク
        """
        from queue import Queue
        from threading import Thread

        # キャンセルフラグをリセット
        self._reset_cancel()

        queue: Queue = Queue()
        finished = {"done": False, "error": None}

        def run_sync_stream():
            try:
                for chunk in self.generate_stream(user_input):
                    # キャンセルチェック
                    if self._cancel_requested:
                        print("[LLM] Stream cancelled by request")
                        break
                    queue.put(("chunk", chunk))
            except Exception as e:
                finished["error"] = e
            finally:
                finished["done"] = True
                queue.put(("done", None))

        thread = Thread(target=run_sync_stream)
        thread.start()

        # タイムアウト設定（30秒）
        timeout_sec = 30
        start_time = asyncio.get_event_loop().time()

        while True:
            # キャンセルチェック
            if self._cancel_requested:
                print("[LLM] Async stream cancelled")
                thread.join(timeout=1.0)  # スレッド終了を最大1秒待機
                return

            # タイムアウトチェック
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_sec:
                print(f"[LLM] Stream timeout after {timeout_sec}s")
                self._cancel_requested = True
                thread.join(timeout=1.0)
                raise TimeoutError(f"LLM stream timeout after {timeout_sec}s")

            await asyncio.sleep(0.01)
            while not queue.empty():
                msg_type, data = queue.get_nowait()
                if msg_type == "chunk":
                    yield data
                elif msg_type == "done":
                    thread.join(timeout=2.0)
                    if finished["error"]:
                        raise finished["error"]
                    return
