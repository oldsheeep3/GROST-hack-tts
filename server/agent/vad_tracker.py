"""VAD Feature Tracker

生PCMから VAD / energy / 発話時間 / 無音時間 を追跡する層
"""
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque

# =============================================================================
# パラメータ
# =============================================================================
FRAME_HOP_MS = 20           # 16kHz なら 320 サンプル/フレーム
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5         # Silero/WebRTC の確信度しきい値

# ターン終了判定
TURN_END_SOFT_MS = 700      # 「そろそろ終わり」判定基準（短すぎると文中ポーズで誤判定）
TURN_END_HARD_MS = 1200     # 「さすがに終わった」判定基準

# Backchannel
BC_MIN_SPEAK_MS = 800       # backchannelを検討し始めるまでの発話時間（短縮: 1500→800）
BC_MIN_SIL_MS = 100         # 相槌に向いたポーズの下限（短縮: 200→100）
BC_MAX_SIL_MS = 500         # 相槌に向いたポーズの上限（短縮: 700→500、応答開始と被らないように）
BC_COOLDOWN_MS = 2500       # 相槌打った後のクールダウン（短縮: 3000→2500）

# Hesitation
HESITATION_MAX_MS = 600     # ためらいのポーズとして許容する無音

# エネルギーゲート（ノイズ対策）
ENERGY_THRESHOLD = 1e4      # これ未満のエネルギーは無音扱い（要調整）


# =============================================================================
# VAD モデル（Silero VAD をラップ）
# =============================================================================
class SileroVADModel:
    """Silero VAD のシングルトンラッパー"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        import torch
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        self.model.eval()
        self._initialized = True
        print("Silero VAD model loaded.")

    def __call__(self, pcm_frame: np.ndarray, sample_rate: int = 16000) -> float:
        """
        pcm_frame: int16 or float32 の 1D array (320 samples @ 16kHz = 20ms)
        Returns: VAD probability (0.0 ~ 1.0)
        """
        import torch

        # int16 → float32 正規化
        if pcm_frame.dtype == np.int16:
            audio = pcm_frame.astype(np.float32) / 32768.0
        else:
            audio = pcm_frame.astype(np.float32)

        # Silero VAD は 512 サンプル以上を期待することがある
        # 320サンプル(20ms)でも動くが、念のためパディング
        if len(audio) < 512:
            audio = np.pad(audio, (0, 512 - len(audio)), mode='constant')

        tensor = torch.from_numpy(audio)

        # Silero VAD は 16000Hz のみサポート
        # 常に16000Hzで呼び出す（呼び出し側で事前にリサンプリング済み前提）
        prob = self.model(tensor, 16000).item()
        return prob


# =============================================================================
# VADFeatureTracker
# =============================================================================
@dataclass
class VADFeatureTracker:
    """VADからの生出力と連続区間の長さを追跡"""

    # VAD の生出力
    vad_prob: float = 0.0
    vad_user: bool = False

    # 連続区間の長さ（ms）
    speak_dur_ms: int = 0       # 連続発話時間
    silence_dur_ms: int = 0     # 連続無音時間

    # 絶対時間（monotonic ms）
    t_now_ms: int = 0
    t_last_voice_ms: Optional[int] = None
    t_first_voice_ms: Optional[int] = None

    # prosody用の簡易特徴（約500ms分 = 25フレーム）
    energy_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=25))
    f0_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=25))

    # backchannel関連
    t_last_bc_ms: Optional[int] = None

    # VADモデル（遅延初期化）
    _vad_model: Optional[SileroVADModel] = field(default=None, repr=False)

    def _get_vad_model(self) -> SileroVADModel:
        if self._vad_model is None:
            self._vad_model = SileroVADModel()
        return self._vad_model

    def update(self, pcm_frame: np.ndarray, t_now_ms: int) -> None:
        """フレームごとの更新

        Args:
            pcm_frame: 320 samples @ 16kHz (20ms) の PCM データ
            t_now_ms: モノトニック時刻 (ms)
        """
        self.t_now_ms = t_now_ms

        # 1. energy 推定（VADより先に計算）
        energy = float(np.mean(pcm_frame.astype(np.float32) ** 2))
        self.energy_hist.append(energy)

        # 2. VAD推定
        vad_model = self._get_vad_model()
        prob = vad_model(pcm_frame)
        self.vad_prob = prob
        raw_vad = prob > VAD_THRESHOLD

        # 3. エネルギーゲート：エネルギーが小さすぎるときは無条件で無音扱い
        if energy < ENERGY_THRESHOLD:
            vad = False
        else:
            vad = raw_vad

        prev_vad = self.vad_user
        self.vad_user = vad

        # 4. f0 推定（簡易版：ゼロ交差率で代用）
        f0 = self._estimate_f0_simple(pcm_frame)
        self.f0_hist.append(f0)

        # 5. 発話 / 無音の継続時間更新
        if vad:
            if not prev_vad:
                # 発話開始
                self.silence_dur_ms = 0
                if self.t_first_voice_ms is None:
                    self.t_first_voice_ms = t_now_ms
            self.speak_dur_ms += FRAME_HOP_MS
            self.t_last_voice_ms = t_now_ms
        else:
            if prev_vad:
                # 無音開始
                self.speak_dur_ms = 0
            self.silence_dur_ms += FRAME_HOP_MS

    def _estimate_f0_simple(self, pcm_frame: np.ndarray) -> float:
        """ゼロ交差率から簡易的にf0を推定"""
        if len(pcm_frame) < 2:
            return 0.0

        # int16 → float
        if pcm_frame.dtype == np.int16:
            audio = pcm_frame.astype(np.float32)
        else:
            audio = pcm_frame

        # ゼロ交差をカウント
        signs = np.sign(audio)
        zero_crossings = np.sum(np.abs(np.diff(signs)) > 0)

        # ゼロ交差率からf0を概算（1秒あたりの交差数 / 2 ≈ 周波数）
        duration_sec = len(pcm_frame) / SAMPLE_RATE
        f0_estimate = zero_crossings / (2 * duration_sec) if duration_sec > 0 else 0.0

        return f0_estimate

    def reset(self) -> None:
        """状態をリセット"""
        self.vad_prob = 0.0
        self.vad_user = False
        self.speak_dur_ms = 0
        self.silence_dur_ms = 0
        self.t_last_voice_ms = None
        self.t_first_voice_ms = None
        self.energy_hist.clear()
        self.f0_hist.clear()
        # t_last_bc_ms はリセットしない（クールダウン維持）

    def get_recent_energy_trend(self) -> Optional[str]:
        """直近のエネルギートレンドを取得

        Returns:
            "falling": 下降傾向
            "rising": 上昇傾向
            "stable": 安定
            None: データ不足
        """
        if len(self.energy_hist) < 3:
            return None

        e1, e2, e3 = list(self.energy_hist)[-3:]

        if e1 > e2 > e3:
            return "falling"
        elif e1 < e2 < e3:
            return "rising"
        else:
            return "stable"
