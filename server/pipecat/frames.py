"""Custom Frame types for the voice agent pipeline.

Design Philosophy:
- VAD is the top-level state machine that governs everything below
- TurnManager receives VAD state and emits turn events
- Downstream processors (STT, LLM, TTS) react to turn events
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from pipecat.frames.frames import (
    Frame,
    DataFrame,
    ControlFrame,
    SystemFrame,
)


# =============================================================================
# VAD State Frames (from VADFeatureTracker)
# =============================================================================

@dataclass
class VADStateFrame(ControlFrame):
    """VAD state update frame - emitted every audio frame.

    This is the primary control signal from the VAD state machine.
    """
    vad_active: bool = False           # Is voice detected
    speak_dur_ms: int = 0              # Continuous speech duration
    silence_dur_ms: int = 0            # Continuous silence duration
    energy_trend: str = "stable"       # "rising", "falling", "stable"
    t_audio_ms: int = 0                # Current audio timestamp


@dataclass
class VADVoiceStartFrame(ControlFrame):
    """Voice activity started."""
    t_ms: int = 0


@dataclass
class VADVoiceEndFrame(ControlFrame):
    """Voice activity ended."""
    t_ms: int = 0
    duration_ms: int = 0


# =============================================================================
# Turn Event Frames (from TurnManager)
# =============================================================================

class TurnEventType(Enum):
    """Turn event types matching existing EventType."""
    USER_START = auto()
    USER_END_SOFT = auto()
    USER_END_HARD = auto()
    BC_WINDOW = auto()
    AGENT_START = auto()
    AGENT_STOP = auto()


@dataclass
class TurnEventFrame(ControlFrame):
    """Turn state change event."""
    event_type: TurnEventType = TurnEventType.USER_START
    t_ms: int = 0
    payload: dict = field(default_factory=dict)


@dataclass
class UserStartFrame(ControlFrame):
    """User started speaking - triggers STT utterance start."""
    t_ms: int = 0


@dataclass
class UserEndSoftFrame(ControlFrame):
    """User likely finished speaking - can trigger early LLM."""
    t_ms: int = 0
    text: str = ""


@dataclass
class UserEndHardFrame(ControlFrame):
    """User definitely finished speaking - triggers LLM response."""
    t_ms: int = 0
    text: str = ""


@dataclass
class BackchannelWindowFrame(ControlFrame):
    """Backchannel opportunity window detected."""
    t_ms: int = 0


@dataclass
class AgentInterruptFrame(ControlFrame):
    """User interrupted agent speech."""
    t_ms: int = 0
    reason: str = "user_speaking"


# =============================================================================
# STT Frames
# =============================================================================

@dataclass
class STTPartialFrame(DataFrame):
    """Partial STT result."""
    text: str = ""
    is_final: bool = False
    t_ms: int = 0


@dataclass
class STTFinalFrame(DataFrame):
    """Final STT result for an utterance."""
    text: str = ""
    t_ms: int = 0


# =============================================================================
# LLM Frames
# =============================================================================

@dataclass
class LLMStartFrame(ControlFrame):
    """LLM generation started."""
    user_text: str = ""


@dataclass
class LLMDeltaFrame(DataFrame):
    """LLM streaming delta."""
    delta: str = ""


@dataclass
class LLMSentenceFrame(DataFrame):
    """Complete sentence from LLM (after segmentation)."""
    sentence: str = ""


@dataclass
class LLMEndFrame(ControlFrame):
    """LLM generation completed."""
    full_text: str = ""


# =============================================================================
# TTS Frames
# =============================================================================

@dataclass
class TTSStartFrame(ControlFrame):
    """TTS synthesis started for a sentence."""
    text: str = ""


@dataclass
class TTSDoneFrame(ControlFrame):
    """TTS synthesis completed."""
    text: str = ""
    duration_ms: int = 0


# =============================================================================
# Backchannel Frames
# =============================================================================

@dataclass
class BackchannelAudioFrame(DataFrame):
    """Pre-cached backchannel audio to play."""
    phrase: str = ""
    audio: bytes = b""
    sample_rate: int = 16000


# =============================================================================
# Session Control Frames
# =============================================================================

@dataclass
class SessionConfigFrame(ControlFrame):
    """Session configuration from client."""
    sample_rate: int = 16000
    config: dict = field(default_factory=dict)


@dataclass
class AgentSpeakingFrame(SystemFrame):
    """Indicates agent is currently speaking (for interrupt detection)."""
    is_speaking: bool = False
    playback_end_time_ms: float = 0
