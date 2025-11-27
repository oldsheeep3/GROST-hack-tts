"""TTS Processor - wraps TTSService for pipecat pipeline.

Handles Style-Bert-VITS2 TTS synthesis.
"""
import asyncio
import numpy as np
from typing import Optional

from pipecat.frames.frames import Frame, AudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from server.agent.services.tts_service import TTSService
from server.pipecat.frames import (
    LLMSentenceFrame,
    TTSStartFrame,
    TTSDoneFrame,
    AgentInterruptFrame,
    AgentSpeakingFrame,
)


class TTSProcessor(FrameProcessor):
    """Processes LLM sentences through TTS.

    Design:
    - Receives LLMSentenceFrame to trigger TTS synthesis
    - Runs synthesis in thread pool (CPU/GPU intensive)
    - Emits AudioRawFrame with synthesized audio
    - Handles AgentInterruptFrame to stop synthesis
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tts: Optional[TTSService] = None
        self._is_synthesizing = False

    async def start(self):
        """Initialize TTS service."""
        self._tts = TTSService()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMSentenceFrame):
            # Synthesize sentence
            if frame.sentence and frame.sentence.strip():
                await self._synthesize(frame.sentence)
            await self.push_frame(frame)

        elif isinstance(frame, AgentInterruptFrame):
            # Stop synthesis
            self._is_synthesizing = False
            await self.push_frame(frame)

        else:
            # Pass through other frames
            await self.push_frame(frame)

    async def _synthesize(self, sentence: str):
        """Synthesize sentence and emit audio frames."""
        if not self._tts:
            return

        self._is_synthesizing = True
        await self.push_frame(TTSStartFrame(text=sentence))

        try:
            # Run synthesis in thread pool
            loop = asyncio.get_event_loop()
            sr, audio = await loop.run_in_executor(
                None, lambda: self._tts.synthesize(sentence)
            )

            if not self._is_synthesizing:
                return

            # Calculate duration
            duration_ms = len(audio) * 1000 // sr

            # Notify that agent is speaking
            import time
            current_time_ms = time.perf_counter() * 1000
            await self.push_frame(AgentSpeakingFrame(
                is_speaking=True,
                playback_end_time_ms=current_time_ms + duration_ms + 100
            ))

            # Convert to int16 bytes for AudioRawFrame
            if audio.dtype != np.int16:
                # Normalize float to int16
                if audio.dtype == np.float32 or audio.dtype == np.float64:
                    audio = (audio * 32767).astype(np.int16)
                else:
                    audio = audio.astype(np.int16)

            # Emit audio frame
            await self.push_frame(AudioRawFrame(
                audio=audio.tobytes(),
                sample_rate=sr,
                num_channels=1
            ))

            await self.push_frame(TTSDoneFrame(text=sentence, duration_ms=duration_ms))

        except Exception as e:
            print(f"[TTS] Error: {e}")
        finally:
            self._is_synthesizing = False

    def reset(self):
        """Reset TTS state."""
        self._is_synthesizing = False
