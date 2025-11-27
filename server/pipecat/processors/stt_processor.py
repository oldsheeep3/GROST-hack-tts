"""STT Processor - wraps DeepgramSTTService for pipecat pipeline.

Handles Deepgram streaming ASR integration.
"""
import asyncio
import numpy as np
from typing import Optional

from pipecat.frames.frames import Frame, AudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from server.agent.services.stt_service import DeepgramSTTService
from server.pipecat.frames import (
    UserStartFrame,
    UserEndHardFrame,
    STTPartialFrame,
    STTFinalFrame,
)


class STTProcessor(FrameProcessor):
    """Processes audio frames through Deepgram STT.

    Design:
    - Receives AudioRawFrame and sends to Deepgram
    - Receives UserStartFrame to start new utterance
    - Receives UserEndHardFrame to finalize utterance
    - Emits STTPartialFrame for interim results
    - Emits STTFinalFrame for final results
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stt: Optional[DeepgramSTTService] = None
        self._started = False
        self._frame_count = 0

    async def start(self):
        """Initialize STT client."""
        self._stt = DeepgramSTTService()
        await self._stt.start()
        self._started = True

    async def stop(self):
        """Stop STT client."""
        if self._stt:
            await self._stt.stop()
        self._started = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if not self._started or not self._stt:
            await self.push_frame(frame)
            return

        if isinstance(frame, AudioRawFrame):
            # Send audio to Deepgram
            pcm_np = np.frombuffer(frame.audio, dtype=np.int16)
            t_audio_ms = self._frame_count * 20  # 20ms frames
            self._frame_count += 1

            await self._stt.push_frame(pcm_np, t_audio_ms)

            # Check for partial results
            partials = await self._stt.get_all_partials()
            for p in partials:
                text = p.get("text", "")
                is_final = p.get("is_final", False)
                if text:
                    await self.push_frame(STTPartialFrame(
                        text=text,
                        is_final=is_final,
                        t_ms=t_audio_ms
                    ))

            # Pass through audio
            await self.push_frame(frame)

        elif isinstance(frame, UserStartFrame):
            # Start new utterance
            self._stt.start_utterance()
            await self.push_frame(frame)

        elif isinstance(frame, UserEndHardFrame):
            # Finalize utterance and get final text
            final_text = await self._stt.end_utterance(timeout_ms=800)
            if final_text:
                await self.push_frame(STTFinalFrame(
                    text=final_text,
                    t_ms=frame.t_ms
                ))
            # Update UserEndHardFrame with final text if available
            if final_text:
                frame = UserEndHardFrame(t_ms=frame.t_ms, text=final_text)
            await self.push_frame(frame)

        else:
            # Pass through other frames
            await self.push_frame(frame)

    def reset(self):
        """Reset for new session."""
        self._frame_count = 0
