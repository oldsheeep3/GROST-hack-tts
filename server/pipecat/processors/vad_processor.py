"""VAD State Processor - wraps VADFeatureTracker for pipecat pipeline.

This is the top-level state machine that governs the entire pipeline.
All audio frames pass through here first, and VAD state is emitted
as control frames for downstream processors.
"""
import numpy as np
from typing import Optional

from pipecat.frames.frames import Frame, AudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from server.agent.vad_tracker import VADFeatureTracker, FRAME_HOP_MS
from server.pipecat.frames import (
    VADStateFrame,
    VADVoiceStartFrame,
    VADVoiceEndFrame,
)


class VADStateProcessor(FrameProcessor):
    """Processes audio frames through VADFeatureTracker and emits state frames.

    Design:
    - Receives AudioRawFrame from transport
    - Updates VADFeatureTracker with each frame
    - Emits VADStateFrame after each update (always)
    - Emits VADVoiceStartFrame / VADVoiceEndFrame on transitions
    - Passes through AudioRawFrame to downstream (for STT)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tracker = VADFeatureTracker()
        self._frame_count = 0
        self._prev_vad_active = False

    @property
    def tracker(self) -> VADFeatureTracker:
        """Access to underlying tracker for TurnControlProcessor."""
        return self._tracker

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            # Convert audio bytes to numpy int16
            pcm_np = np.frombuffer(frame.audio, dtype=np.int16)

            # Calculate timestamp
            t_audio_ms = self._frame_count * FRAME_HOP_MS
            self._frame_count += 1

            # Update VAD tracker
            self._tracker.update(pcm_np, t_audio_ms)

            # Detect transitions
            vad_active = self._tracker.vad_user
            if vad_active and not self._prev_vad_active:
                # Voice started
                await self.push_frame(VADVoiceStartFrame(t_ms=t_audio_ms))
            elif not vad_active and self._prev_vad_active:
                # Voice ended
                await self.push_frame(VADVoiceEndFrame(
                    t_ms=t_audio_ms,
                    duration_ms=self._tracker.speak_dur_ms
                ))

            self._prev_vad_active = vad_active

            # Always emit state frame for downstream processors
            state_frame = VADStateFrame(
                vad_active=vad_active,
                speak_dur_ms=self._tracker.speak_dur_ms,
                silence_dur_ms=self._tracker.silence_dur_ms,
                energy_trend=self._tracker.get_recent_energy_trend(),
                t_audio_ms=t_audio_ms,
            )
            await self.push_frame(state_frame)

            # Pass through audio frame for STT
            await self.push_frame(frame)
        else:
            # Pass through other frames
            await self.push_frame(frame)

    def reset(self):
        """Reset VAD state for new session."""
        self._tracker.reset()
        self._frame_count = 0
        self._prev_vad_active = False
