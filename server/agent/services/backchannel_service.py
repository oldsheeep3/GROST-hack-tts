"""相槌サービス - 事前生成された音声ファイルを読み込んで再生"""
import random
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np


# 音声ファイルディレクトリ
ASSETS_DIR = Path(__file__).parent.parent.parent / "assets" / "backchannel"

# クールダウン設定（ミリ秒）
BC_COOLDOWN_MS = 2500


class BackchannelService:
    """相槌音声のキャッシュと選択を管理するサービス"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # キャッシュ: [(phrase, sample_rate, audio_array), ...]
        self._cache: List[Tuple[str, int, np.ndarray]] = []
        self._last_used_time_ms: float = 0
        self._initialized = True

    def load_from_files(self) -> None:
        """事前生成された音声ファイルを読み込む"""
        manifest_path = ASSETS_DIR / "manifest.txt"

        if not manifest_path.exists():
            print(f"[BackchannelService] WARNING: {manifest_path} not found")
            print("[BackchannelService] Run: python scripts/generate_backchannel_audio.py")
            return

        print("[BackchannelService] Loading pre-generated audio files...")
        with open(manifest_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                filename, phrase = line.split('\t')
                filepath = ASSETS_DIR / filename

                if not filepath.exists():
                    print(f"  WARNING: {filepath} not found")
                    continue

                data = np.load(filepath)
                audio = data['audio']
                sr = int(data['sample_rate'])

                self._cache.append((phrase, sr, audio))
                duration_ms = len(audio) * 1000 // sr
                print(f"  - '{phrase}': {duration_ms}ms")

        print(f"[BackchannelService] Loaded {len(self._cache)} phrases")

    def warmup(self, tts_service=None) -> None:
        """互換性のためのメソッド（ファイル読み込みを行う）"""
        self.load_from_files()

    def can_play(self) -> bool:
        """クールダウン中でなければTrue"""
        current_ms = time.perf_counter() * 1000
        return (current_ms - self._last_used_time_ms) >= BC_COOLDOWN_MS

    def get_random_audio(self) -> Optional[Tuple[str, int, np.ndarray]]:
        """ランダムな相槌音声を取得（クールダウン更新込み）

        Returns:
            (phrase, sample_rate, audio_array) or None if cooldown or no cache
        """
        if not self.can_play():
            return None

        if not self._cache:
            return None

        phrase, sr, audio = random.choice(self._cache)
        self._last_used_time_ms = time.perf_counter() * 1000
        return (phrase, sr, audio)

    @property
    def phrases(self) -> List[str]:
        return [item[0] for item in self._cache]

    @property
    def loaded(self) -> bool:
        return len(self._cache) > 0
