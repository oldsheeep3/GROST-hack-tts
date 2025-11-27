"""文セグメント用ユーティリティ"""
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class STTAccumulator:
    """STTからの部分テキストを蓄積し、LLM開始タイミングを判断"""
    buf: str = ""
    llm_started: bool = False
    first_token_time: Optional[float] = None

    # 閾値（調整可能）
    MIN_CHARS_FOR_LLM: int = 10  # 文字数閾値（緩めに設定）
    MAX_WAIT_TIME_SEC: float = 0.7  # 最初のトークンから待つ最大時間

    def update(self, delta: str) -> None:
        """Qwenから来たpartial（差分）を足す"""
        if self.first_token_time is None and delta:
            self.first_token_time = time.perf_counter()
        self.buf += delta

    @property
    def text(self) -> str:
        return self.buf

    def ready_for_llm(self) -> bool:
        """LLMをスタートさせるかの条件

        以下のいずれかを満たしたらTrue:
        1. 文字数がMIN_CHARS_FOR_LLM以上
        2. 「。」「？」「！」「、」が含まれる
        3. 最初のトークンからMAX_WAIT_TIME_SEC秒経過
        """
        txt = self.buf

        # 1. 文字数閾値
        if len(txt) >= self.MIN_CHARS_FOR_LLM:
            return True

        # 2. 句読点チェック（読点「、」も含める）
        if any(c in txt for c in "。？！、"):
            return True

        # 3. 経過時間チェック
        if self.first_token_time is not None:
            elapsed = time.perf_counter() - self.first_token_time
            if elapsed >= self.MAX_WAIT_TIME_SEC:
                return True

        return False


@dataclass
class SentenceSegmenter:
    """LLM出力を文単位で切るセグメンタ"""
    buf: str = ""
    committed: List[str] = field(default_factory=list)

    def push(self, delta: str) -> List[str]:
        """LLM streamingからのdeltaを受け取り、完成した文をリストで返す"""
        self.buf += delta
        new_sentences: List[str] = []

        # 「。」「？」「！」で切る
        for delimiter in ["。", "？", "！", "?", "!"]:
            while delimiter in self.buf:
                idx = self.buf.index(delimiter)
                sentence = self.buf[:idx + 1]
                self.buf = self.buf[idx + 1:]

                sentence = sentence.strip()
                if len(sentence) < 4:
                    # 短すぎる文は次に繋げる
                    self.buf = sentence + self.buf
                    continue

                new_sentences.append(sentence)
                self.committed.append(sentence)

        return new_sentences

    def flush_last(self) -> List[str]:
        """ストリーム終了時に最後の文を吐く"""
        s = self.buf.strip()
        self.buf = ""
        if s and len(s) >= 2:
            self.committed.append(s)
            return [s]
        return []
