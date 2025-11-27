"""Turn Control Processor - wraps TurnManager for pipecat pipeline.

Receives VAD state frames and emits turn control events.
This processor is the decision maker for conversation flow.
"""
from typing import Optional

from pipecat.frames.frames import Frame, AudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from server.agent.turn_manager import TurnManager, EventType, ConvState
from server.agent.vad_tracker import VADFeatureTracker
from server.pipecat.frames import (
    VADStateFrame,
    STTPartialFrame,
    STTFinalFrame,
    UserStartFrame,
    UserEndSoftFrame,
    UserEndHardFrame,
    BackchannelWindowFrame,
    AgentInterruptFrame,
    AgentSpeakingFrame,
)


class TurnControlProcessor(FrameProcessor):
    """Processes VAD state and STT results to manage conversation turns.

    Design:
    - Receives VADStateFrame from VADStateProcessor
    - Receives STTPartialFrame from STT processor
    - Maintains TurnManager state machine
    - Emits turn control frames (UserStart, UserEndSoft, UserEndHard, etc.)
    - Tracks agent speaking state for interrupt detection
    """

    def __init__(self, vad_tracker: Optional[VADFeatureTracker] = None, **kwargs):
        super().__init__(**kwargs)
        self._turn_manager = TurnManager()
        # Link VAD tracker if provided (for shared state)
        if vad_tracker:
            self._turn_manager.tracker = vad_tracker
        self._is_agent_speaking = False
        self._playback_end_time_ms: float = 0

    @property
    def turn_manager(self) -> TurnManager:
        """Access to underlying turn manager."""
        return self._turn_manager

    def set_vad_tracker(self, tracker: VADFeatureTracker):
        """Set VAD tracker (called during pipeline setup)."""
        self._turn_manager.tracker = tracker

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADStateFrame):
            await self._handle_vad_state(frame)
        elif isinstance(frame, STTPartialFrame):
            await self._handle_stt_partial(frame)
        elif isinstance(frame, STTFinalFrame):
            await self._handle_stt_final(frame)
        elif isinstance(frame, AgentSpeakingFrame):
            # Update agent speaking state
            self._is_agent_speaking = frame.is_speaking
            self._playback_end_time_ms = frame.playback_end_time_ms
            await self.push_frame(frame)
        else:
            # Pass through other frames
            await self.push_frame(frame)

    async def _handle_vad_state(self, frame: VADStateFrame):
        """Handle VAD state update and run turn manager logic."""
        t_audio_ms = frame.t_audio_ms

        # Update turn manager (VAD already updated by VADStateProcessor)
        events = self._turn_manager.update_without_vad(t_audio_ms, "")

        # Emit turn events
        for ev_type, payload in events:
            await self._emit_turn_event(ev_type, t_audio_ms, payload)

        # Check for LLM trigger
        if self._turn_manager.should_start_llm() and not self._turn_manager.llm_started:
            self._turn_manager.mark_llm_started()
            # LLM will be started by downstream processor when it receives UserEndHard

        # Pass through VAD state for downstream processors
        await self.push_frame(frame)

    async def _handle_stt_partial(self, frame: STTPartialFrame):
        """Handle STT partial result."""
        # Accumulate in turn manager
        if frame.text:
            self._turn_manager.stt_text += frame.text

        # Pass through for logging/display
        await self.push_frame(frame)

    async def _handle_stt_final(self, frame: STTFinalFrame):
        """Handle STT final result."""
        # Update turn manager with final text
        if frame.text:
            self._turn_manager.stt_text = frame.text

        # Pass through
        await self.push_frame(frame)

    async def _emit_turn_event(self, ev_type: EventType, t_ms: int, payload: dict):
        """Convert TurnManager event to frame and emit."""
        if ev_type == EventType.USER_START:
            await self.push_frame(UserStartFrame(t_ms=t_ms))
        elif ev_type == EventType.USER_END_SOFT:
            text = self._turn_manager.stt_text or payload.get("text", "")
            await self.push_frame(UserEndSoftFrame(t_ms=t_ms, text=text))
        elif ev_type == EventType.USER_END_HARD:
            text = self._turn_manager.stt_text or payload.get("text", "")
            await self.push_frame(UserEndHardFrame(t_ms=t_ms, text=text))
        elif ev_type == EventType.BC_WINDOW:
            await self.push_frame(BackchannelWindowFrame(t_ms=t_ms))
        elif ev_type == EventType.AGENT_STOP_SPEAKING:
            reason = payload.get("reason", "user_speaking")
            await self.push_frame(AgentInterruptFrame(t_ms=t_ms, reason=reason))

    def notify_agent_speaking(self, is_speaking: bool, playback_end_ms: float = 0):
        """Called when agent starts/stops speaking."""
        self._is_agent_speaking = is_speaking
        self._playback_end_time_ms = playback_end_ms

    def notify_llm_done(self):
        """Called when LLM+TTS response is complete."""
        self._turn_manager.reset_for_new_turn()
        self._is_agent_speaking = False
        self._playback_end_time_ms = 0

    def reset(self):
        """Reset turn state for new session."""
        self._turn_manager.reset_for_new_turn()
        self._is_agent_speaking = False
        self._playback_end_time_ms = 0
