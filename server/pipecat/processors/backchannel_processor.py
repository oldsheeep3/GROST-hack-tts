"""Backchannel Processor - wraps BackchannelService for pipecat pipeline.

Handles backchannel audio injection.
"""
import base64
import numpy as np
from typing import Optional

from pipecat.frames.frames import Frame, AudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from server.agent.services.backchannel_service import BackchannelService
from server.agent.services.tts_service import TTSService
from server.pipecat.frames import (
    BackchannelWindowFrame,
    BackchannelAudioFrame,
)


class BackchannelProcessor(FrameProcessor):
    """Processes backchannel opportunities.

    Design:
    - Receives BackchannelWindowFrame from TurnControlProcessor
    - Retrieves cached backchannel audio
    - Emits BackchannelAudioFrame (or AudioRawFrame) for playback
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._backchannel = BackchannelService()
        self._initialized = False

    async def start(self):
        """Initialize backchannel service with cached audio."""
        # Warmup loads cached audio files
        tts = TTSService()  # Needed for potential regeneration
        self._backchannel.warmup(tts)
        self._initialized = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BackchannelWindowFrame):
            # Try to get backchannel audio
            if self._initialized:
                result = self._backchannel.get_random_audio()
                if result:
                    phrase, sr, audio = result
                    print(f"[Backchannel] Playing: '{phrase}'")

                    # Convert to int16 if needed
                    if audio.dtype != np.int16:
                        if audio.dtype == np.float32 or audio.dtype == np.float64:
                            audio = (audio * 32767).astype(np.int16)
                        else:
                            audio = audio.astype(np.int16)

                    # Emit backchannel audio frame
                    await self.push_frame(BackchannelAudioFrame(
                        phrase=phrase,
                        audio=audio.tobytes(),
                        sample_rate=sr
                    ))

            await self.push_frame(frame)

        else:
            # Pass through other frames
            await self.push_frame(frame)
