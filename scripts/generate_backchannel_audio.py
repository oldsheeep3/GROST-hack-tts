#!/usr/bin/env python3
"""相槌音声ファイルを事前生成するスクリプト

使い方:
  python scripts/generate_backchannel_audio.py

生成先:
  server/assets/backchannel/*.npy (16kHz int16 PCM)
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from server.agent.services.tts_service import TTSService

# 相槌フレーズ
BACKCHANNEL_PHRASES = [
    "うんうん",
    "なるほどね",
    "へー",
    "ふーん",
    "そっか",
    "あー",
    "うん",
]

OUTPUT_DIR = PROJECT_ROOT / "server" / "assets" / "backchannel"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Initializing TTS service...")
    tts = TTSService()

    print(f"\nGenerating {len(BACKCHANNEL_PHRASES)} backchannel audio files...")

    for phrase in BACKCHANNEL_PHRASES:
        print(f"  Synthesizing: '{phrase}'")
        sr, audio = tts.synthesize(phrase)

        # ファイル名（日本語はハッシュで）
        filename = f"{hash(phrase) & 0xFFFFFFFF:08x}.npy"
        filepath = OUTPUT_DIR / filename

        # メタデータ付きで保存
        np.savez(
            filepath.with_suffix('.npz'),
            audio=audio.astype(np.int16),
            sample_rate=sr,
            phrase=phrase
        )

        duration_ms = len(audio) * 1000 // sr
        print(f"    -> {filepath.with_suffix('.npz')} ({duration_ms}ms)")

    # マニフェストファイル生成
    manifest_path = OUTPUT_DIR / "manifest.txt"
    with open(manifest_path, 'w') as f:
        for phrase in BACKCHANNEL_PHRASES:
            filename = f"{hash(phrase) & 0xFFFFFFFF:08x}.npz"
            f.write(f"{filename}\t{phrase}\n")

    print(f"\nManifest: {manifest_path}")
    print("Done!")


if __name__ == "__main__":
    main()
