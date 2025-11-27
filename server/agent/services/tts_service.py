import numpy as np
from pathlib import Path
from style_bert_vits2.tts_model import TTSModel
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages

from server.config import BERT_MODEL_PATH, TTS_MODEL_DIR, TTS_DEVICE, TTS_DEFAULT_STYLE


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
        self.model = TTSModel(
            model_path=TTS_MODEL_DIR / "jvnv-F1-jp_e160_s14000.safetensors",
            config_path=TTS_MODEL_DIR / "config.json",
            style_vec_path=TTS_MODEL_DIR / "style_vectors.npy",
            device=TTS_DEVICE
        )

        # ウォームアップ
        print("Warming up TTS model...")
        self.model.infer("テスト", style=TTS_DEFAULT_STYLE)

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
        sr, audio = self.model.infer(text, style=style)
        return sr, audio
