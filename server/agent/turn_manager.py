"""Turn Manager

VADFeatureTracker の出力 + STT状態 を見て、
state遷移とイベント発火を行う層
"""
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from server.agent.vad_tracker import (
    VADFeatureTracker,
    FRAME_HOP_MS,
    TURN_END_SOFT_MS,
    TURN_END_HARD_MS,
    BC_MIN_SPEAK_MS,
    BC_MIN_SIL_MS,
    BC_MAX_SIL_MS,
    BC_COOLDOWN_MS,
    HESITATION_MAX_MS,
)

# USER_START を出す前に必要な最低発話長（ノイズ対策）
# ★ 短い発話も受け付けるために緩和（STTで内容判定する方がマシ）
MIN_UTTER_MS = 60  # 60ms (3フレーム) 未満の VAD は無視


# =============================================================================
# イベント定義
# =============================================================================
class EventType(Enum):
    USER_START = auto()           # ユーザ発話開始
    USER_END_SOFT = auto()        # 終わりかけ
    USER_END_HARD = auto()        # 終了確定
    BC_WINDOW = auto()            # 相槌の窓
    AGENT_START_FAST = auto()     # 前置き再生開始
    AGENT_START_MAIN = auto()     # 本編TTS開始（最初の文）
    AGENT_STOP_SPEAKING = auto()  # エージェント発話停止


# =============================================================================
# 会話状態定義
# =============================================================================
class ConvState(Enum):
    IDLE = auto()                 # ユーザもエージェントも喋ってない
    LISTENING = auto()            # ユーザ発話中（or 継続しそう）
    HESITATION_GAP = auto()       # 短いポーズ中（まだ続きそう）
    BACKCHANNEL_PENDING = auto()  # 相槌を打つことが決まった状態
    PLANNING_FAST = auto()        # ターン終端直後、前置き再生中
    SPEAKING_MAIN = auto()        # 本編TTS再生中


# =============================================================================
# TurnManager
# =============================================================================
@dataclass
class TurnManager:
    """ターン管理ステートマシン"""

    # 現在の状態
    state: ConvState = ConvState.IDLE

    # VAD情報
    tracker: VADFeatureTracker = field(default_factory=VADFeatureTracker)

    # ターン境界タイムスタンプ
    t_user_start_ms: Optional[int] = None
    t_user_end_soft_ms: Optional[int] = None
    t_user_end_hard_ms: Optional[int] = None

    # backchannel 関連
    bc_pending: bool = False

    # STT関連
    stt_text: str = ""
    stt_first_partial_time_ms: Optional[int] = None

    # LLM / TTS 管理
    llm_started: bool = False
    t_llm_start_ms: Optional[int] = None
    t_first_audio_ready_ms: Optional[int] = None

    # 前回の状態（ログ用）
    _prev_state: ConvState = field(default=ConvState.IDLE, repr=False)

    def update(
        self,
        pcm_frame: np.ndarray,
        t_now_ms: int,
        stt_partial: str = ""
    ) -> List[Tuple[EventType, Dict[str, Any]]]:
        """フレームごとの更新

        Args:
            pcm_frame: 生の20msフレーム (320 samples @ 16kHz)
            t_now_ms: モノトニック時刻 (ms)
            stt_partial: 現時点でのSTT partial (差分)

        Returns:
            発火したイベントのリスト [(EventType, payload), ...]
        """
        events: List[Tuple[EventType, Dict[str, Any]]] = []

        # 1. VAD更新
        self.tracker.update(pcm_frame, t_now_ms)

        # 2. STTテキスト更新と状態遷移
        return self._process_state(t_now_ms, stt_partial)

    def update_without_vad(
        self,
        t_now_ms: int,
        stt_partial: str = "",
    ) -> List[Tuple[EventType, Dict[str, Any]]]:
        """VAD更新なしでステートマシンを更新（VADは既に更新済みの場合）"""
        return self._process_state(t_now_ms, stt_partial)

    def _process_state(
        self,
        t_now_ms: int,
        stt_partial: str = "",
    ) -> List[Tuple[EventType, Dict[str, Any]]]:
        """状態遷移処理の共通部分"""
        events: List[Tuple[EventType, Dict[str, Any]]] = []

        # STTテキスト更新
        if stt_partial:
            if self.stt_first_partial_time_ms is None:
                self.stt_first_partial_time_ms = t_now_ms
            self.stt_text += stt_partial

        # 現ステートに応じた処理
        if self.state == ConvState.IDLE:
            events += self._update_idle()
        elif self.state == ConvState.LISTENING:
            events += self._update_listening()
        elif self.state == ConvState.HESITATION_GAP:
            events += self._update_hesitation()
        elif self.state == ConvState.BACKCHANNEL_PENDING:
            events += self._update_backchannel_pending()
        elif self.state == ConvState.PLANNING_FAST:
            events += self._update_planning_fast()
        elif self.state == ConvState.SPEAKING_MAIN:
            events += self._update_speaking_main()

        # 状態遷移ログ
        if self.state != self._prev_state:
            self._log_state_change(self._prev_state, self.state, events)
            self._prev_state = self.state

        return events

    # =========================================================================
    # 各ステートの更新ロジック
    # =========================================================================

    def _update_idle(self) -> List[Tuple[EventType, Dict[str, Any]]]:
        """IDLE: ユーザもエージェントも喋ってない"""
        ev = []
        tr = self.tracker

        if tr.vad_user:
            # 最低発話長に達していなければノイズ扱いで無視
            if tr.speak_dur_ms < MIN_UTTER_MS:
                return ev

            # ここまで来たら「ちゃんと喋っている」とみなす
            self.state = ConvState.LISTENING
            # start 時刻は first_voice_ms を使う方が自然
            self.t_user_start_ms = tr.t_first_voice_ms or tr.t_now_ms
            self.stt_text = ""
            self.stt_first_partial_time_ms = None
            self.llm_started = False
            self.t_llm_start_ms = None
            self.t_user_end_soft_ms = None
            self.t_user_end_hard_ms = None
            ev.append((EventType.USER_START, {
                "t": tr.t_now_ms,
            }))

        return ev

    def _update_listening(self) -> List[Tuple[EventType, Dict[str, Any]]]:
        """LISTENING: ユーザ発話中"""
        ev = []
        tr = self.tracker
        t = tr.t_now_ms

        # 1. Backchannel window 判定 (HESITATION より先に判定)
        #    200〜700ms の無音で相槌を入れたいが、HESITATIONが先だと200-600msで抜けてしまう
        if self._is_bc_window():
            self.state = ConvState.BACKCHANNEL_PENDING
            self.tracker.t_last_bc_ms = t
            ev.append((EventType.BC_WINDOW, {"t": t}))
            return ev

        # 2. HESITATION判定（まだ続きそうな短いポーズ）
        if not tr.vad_user and 0 < tr.silence_dur_ms <= HESITATION_MAX_MS:
            self.state = ConvState.HESITATION_GAP
            return ev

        # 3. TURN_END_HARD 判定（終了確定）- SOFTより先に判定
        if self._is_turn_end_hard():
            self.state = ConvState.PLANNING_FAST
            self.t_user_end_hard_ms = t
            ev.append((EventType.USER_END_HARD, {
                "t": t,
                "stt_final": self.stt_text,
            }))
            ev.append((EventType.AGENT_START_FAST, {
                "t": t,
            }))
            return ev

        # 4. TURN_END_SOFT 判定（終わりかけ）
        if self._is_turn_end_soft():
            if self.t_user_end_soft_ms is None:
                self.t_user_end_soft_ms = t
                ev.append((EventType.USER_END_SOFT, {
                    "t": t,
                    "stt_preview": self.stt_text,
                }))

        return ev

    def _update_hesitation(self) -> List[Tuple[EventType, Dict[str, Any]]]:
        """HESITATION_GAP: 短いポーズ中（まだ続きそう）"""
        ev = []
        tr = self.tracker

        if tr.vad_user:
            # 再開 → LISTENING に戻す
            self.state = ConvState.LISTENING
            return ev

        if tr.silence_dur_ms > HESITATION_MAX_MS:
            # ためらいの範囲を超えた → LISTENING に戻してsoft/hard判定
            self.state = ConvState.LISTENING
            # ここで即座に LISTENING の判定を実行
            ev += self._update_listening()

        return ev

    def _update_backchannel_pending(self) -> List[Tuple[EventType, Dict[str, Any]]]:
        """BACKCHANNEL_PENDING: 相槌を打つことが決まった状態"""
        ev = []
        tr = self.tracker

        # ユーザが急に喋り始めたら LISTENING に戻す
        if tr.vad_user:
            self.state = ConvState.LISTENING
            return ev

        # 相槌WAV が終わった時点で外から notify_backchannel_done() が呼ばれる想定
        return ev

    def _update_planning_fast(self) -> List[Tuple[EventType, Dict[str, Any]]]:
        """PLANNING_FAST: ターン終了確定直後、前置き再生中"""
        ev = []
        tr = self.tracker

        # ユーザが被せてきたら前置きを止めて LISTENING に戻す
        if tr.vad_user:
            ev.append((EventType.AGENT_STOP_SPEAKING, {"reason": "user_interruption"}))
            # ★ 新しいターンを開始するために USER_START も発火
            self.t_user_start_ms = tr.t_now_ms
            self.stt_text = ""
            self.stt_first_partial_time_ms = None
            self.llm_started = False
            self.t_llm_start_ms = None
            self.t_user_end_soft_ms = None
            self.t_user_end_hard_ms = None
            ev.append((EventType.USER_START, {"t": tr.t_now_ms}))
            self.state = ConvState.LISTENING
            # 進行中の LLM/TTS はキャンセルするかどうか外側で判断
            return ev

        # 前置きWAV終了は外から notify_fast_done() が呼ばれる想定
        return ev

    def _update_speaking_main(self) -> List[Tuple[EventType, Dict[str, Any]]]:
        """SPEAKING_MAIN: 本編TTS再生中"""
        ev = []
        tr = self.tracker

        if tr.vad_user:
            # ユーザが被せた → AGENTをフェードアウト
            ev.append((EventType.AGENT_STOP_SPEAKING, {"reason": "user_interruption"}))
            # ★ 新しいターンを開始するために USER_START も発火
            self.t_user_start_ms = tr.t_now_ms
            self.stt_text = ""
            self.stt_first_partial_time_ms = None
            self.llm_started = False
            self.t_llm_start_ms = None
            self.t_user_end_soft_ms = None
            self.t_user_end_hard_ms = None
            ev.append((EventType.USER_START, {"t": tr.t_now_ms}))
            self.state = ConvState.LISTENING
            return ev

        # 本編TTS終了は外から notify_main_done() が呼ばれる想定
        return ev

    # =========================================================================
    # 判定関数
    # =========================================================================

    def _is_bc_window(self) -> bool:
        """相槌の窓かどうか判定"""
        tr = self.tracker
        t = tr.t_now_ms

        # ターンが始まっていない / もう終わっているなら絶対に相槌を出さない
        if self.t_user_start_ms is None:
            return False
        if self.t_user_end_hard_ms is not None:
            return False

        # 現在無音でなければNG
        if tr.vad_user:
            return False

        # 無音が始まったばかりならNG
        if tr.silence_dur_ms == 0:
            return False

        # 実際の発話時間（speak_dur_ms の累積）が一定以上続いた後でなければNG
        # ※ total_user_speech = t - t_user_start_ms だとノイズでも時間だけ経過する問題があった
        if tr.speak_dur_ms < BC_MIN_SPEAK_MS:
            return False

        # 無音時間が相槌に適した範囲かチェック
        if not (BC_MIN_SIL_MS <= tr.silence_dur_ms <= BC_MAX_SIL_MS):
            return False

        # 最近相槌を打ちすぎていないかチェック
        if tr.t_last_bc_ms is not None and t - tr.t_last_bc_ms < BC_COOLDOWN_MS:
            return False

        # energy が直前で下がっているか（終端感）- 条件を緩和
        # "falling" or "stable" を許容（無音中はstableになりやすいため）
        trend = tr.get_recent_energy_trend()
        if trend not in ("falling", "stable"):
            return False

        return True

    def _is_turn_end_soft(self) -> bool:
        """ターン終了（ソフト）かどうか判定"""
        tr = self.tracker

        # 無音が TURN_END_SOFT_MS 超えていなければNG
        if tr.silence_dur_ms < TURN_END_SOFT_MS:
            return False

        # energy が直前で明確に下がっているか
        trend = tr.get_recent_energy_trend()
        if trend != "falling":
            return False

        # STT側のテキストもそれなりに溜まっているか
        if len(self.stt_text) < 10:
            return False

        return True

    def _is_turn_end_hard(self) -> bool:
        """ターン終了（ハード）かどうか判定"""
        tr = self.tracker
        return tr.silence_dur_ms >= TURN_END_HARD_MS

    def should_start_llm(self) -> bool:
        """LLMを起動すべきかどうか判定

        TURN_END_SOFT が立っている、または
        STT が ready_for_llm 条件を満たしている場合に True
        """
        if self.llm_started:
            return False

        # TURN_END_SOFT が一度でも立っている
        if self.t_user_end_soft_ms is not None:
            return True

        # STT が ready_for_llm 条件を満たしている
        if len(self.stt_text) >= 10:
            return True

        # 最初のpartialから一定時間経過
        if self.stt_first_partial_time_ms is not None:
            elapsed = self.tracker.t_now_ms - self.stt_first_partial_time_ms
            if elapsed > 700:
                return True

        return False

    # =========================================================================
    # 外部からの通知
    # =========================================================================

    def notify_backchannel_done(self) -> None:
        """相槌WAV再生完了の通知"""
        if self.state == ConvState.BACKCHANNEL_PENDING:
            self.state = ConvState.LISTENING

    def notify_fast_done(self) -> List[Tuple[EventType, Dict[str, Any]]]:
        """前置きWAV再生完了の通知"""
        ev = []
        if self.state == ConvState.PLANNING_FAST:
            self.state = ConvState.SPEAKING_MAIN
            ev.append((EventType.AGENT_START_MAIN, {
                "t": self.tracker.t_now_ms,
            }))
        return ev

    def notify_main_done(self) -> None:
        """本編TTS再生完了の通知"""
        if self.state == ConvState.SPEAKING_MAIN:
            self.state = ConvState.IDLE

    def mark_llm_started(self) -> None:
        """LLM起動をマーク"""
        self.llm_started = True
        self.t_llm_start_ms = self.tracker.t_now_ms

    def reset_for_new_turn(self) -> None:
        """新しいターンのために状態をリセット（割り込み時などに使用）"""
        self.state = ConvState.IDLE

        self.t_user_start_ms = None
        self.t_user_end_soft_ms = None
        self.t_user_end_hard_ms = None

        self.bc_pending = False

        self.stt_text = ""
        self.stt_first_partial_time_ms = None

        self.llm_started = False
        self.t_llm_start_ms = None
        self.t_first_audio_ready_ms = None

        # VAD側もターン境界でリセット
        self.tracker.reset()
        print(f"[TurnManager] Reset for new turn")

    # =========================================================================
    # ログ出力
    # =========================================================================

    def _log_state_change(
        self,
        prev: ConvState,
        curr: ConvState,
        events: List[Tuple[EventType, Dict[str, Any]]]
    ) -> None:
        """状態遷移のログ出力"""
        t = self.tracker.t_now_ms
        event_names = [e[0].name for e in events]
        print(f"[TurnManager] {t}ms: {prev.name} → {curr.name} | events={event_names}")

    def get_debug_info(self) -> Dict[str, Any]:
        """デバッグ情報を取得"""
        tr = self.tracker
        return {
            "state": self.state.name,
            "t_now_ms": tr.t_now_ms,
            "vad_user": tr.vad_user,
            "vad_prob": f"{tr.vad_prob:.2f}",
            "speak_dur_ms": tr.speak_dur_ms,
            "silence_dur_ms": tr.silence_dur_ms,
            "stt_text": self.stt_text[:30] + "..." if len(self.stt_text) > 30 else self.stt_text,
            "llm_started": self.llm_started,
            "energy_trend": tr.get_recent_energy_trend(),
        }
