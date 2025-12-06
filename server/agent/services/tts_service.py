import sys
import os
import numpy as np
from pathlib import Path

# pyopenjtalkの互換性エラーを事前に回避
os.environ['PYTHONWARNINGS'] = 'ignore'
# numpy.dtypeエラー回避のための環境設定
os.environ['OMP_NUM_THREADS'] = '1'

from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages

from server.config import BERT_MODEL_PATH, TTS_MODEL_DIR, TTS_DEVICE, TTS_DEFAULT_STYLE, TTS_MODEL_NAME


def download_tts_model():
    """Download TTS model from HuggingFace if not exists"""
    if TTS_MODEL_DIR.exists() and (TTS_MODEL_DIR / "config.json").exists():
        return
    
    print(f"[TTS] Model not found at {TTS_MODEL_DIR}, downloading...")
    TTS_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    
    # 利用可能な公開モデル
    repos = [
        "litagin/style_bert_vits2_jvnv",  # JVNV (公式)
        "ayousanz/tsukuyomi-chan-style-bert-vits2-model",  # つくよみちゃん
        "litagin/sbv2_koharune_ami",  # 小春音アミ
    ]
    
    for repo_id in repos:
        try:
            from huggingface_hub import snapshot_download
            print(f"[TTS] Downloading from {repo_id}...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(TTS_MODEL_DIR),
                local_dir_use_symlinks=False
            )
            print(f"[TTS] ✓ Download completed from {repo_id}")
            return
        except Exception as e:
            print(f"[TTS] Failed to download from {repo_id}: {str(e)[:200]}")
            continue
    
    # すべて失敗した場合
    raise RuntimeError(
        f"Failed to download TTS model from any repository.\n"
        f"Please manually download a Style-Bert-VITS2 model and place it in: {TTS_MODEL_DIR}\n"
        f"See: https://github.com/litagin02/Style-Bert-VITS2"
    )


class TTSService:
    """Style-Bert-VITS2をラップしたTTSサービス"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # モデルが存在しない場合は自動ダウンロード
        download_tts_model()

        print("Loading BERT model...")
        bert_models.load_model(
            Languages.JP,
            pretrained_model_name_or_path=str(BERT_MODEL_PATH)
        )
        bert_models.load_tokenizer(
            Languages.JP,
            pretrained_model_name_or_path=str(BERT_MODEL_PATH)
        )

        print("Loading TTS model...")
        # モデルファイルは TTS_MODEL_NAME サブディレクトリにある
        model_files_dir = TTS_MODEL_DIR / TTS_MODEL_NAME
        if not model_files_dir.exists():
            model_files_dir = TTS_MODEL_DIR  # フォールバック
        
        # モデルファイル名を検出
        safetensors_files = list(model_files_dir.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No .safetensors file found in {model_files_dir}")
        
        model_file = safetensors_files[0]
        print(f"Using model: {model_file.name}")
            
        self.model = TTSModel(
            model_path=model_file,
            config_path=model_files_dir / "config.json",
            style_vec_path=model_files_dir / "style_vectors.npy",
            device=TTS_DEVICE
        )

        # ウォームアップ（エラーは無視）
        print("Warming up TTS model...")
        try:
            self.model.infer("テスト", style=TTS_DEFAULT_STYLE)
        except Exception as e:
            print(f"⚠️  TTS warmup failed: {e}")
            print("Continuing without warmup...")

        self._initialized = True
        print("TTS service initialized.")

    def synthesize(self, text: str, style: str = TTS_DEFAULT_STYLE) -> tuple[int, np.ndarray]:
        """テキストから音声を生成

        Args:
            text: 合成するテキスト
            style: スタイル名

        Returns:
            (sample_rate, audio_array) のタプル
        """
        try:
            sr, audio = self.model.infer(text, style=style)
            return sr, audio
        except ValueError as e:
            if "numpy.dtype size changed" in str(e):
                # numpy互換性エラー - フォールバック処理
                print(f"⚠️  numpy compatibility error, retrying...")
                import time
                time.sleep(0.5)
                sr, audio = self.model.infer(text, style=style)
                return sr, audio
            raise
