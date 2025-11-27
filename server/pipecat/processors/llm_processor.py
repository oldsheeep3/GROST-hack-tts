"""LLM Processor - wraps LLMService for pipecat pipeline.

Handles Gemini streaming LLM integration with sentence segmentation.
"""
import asyncio
from typing import Optional

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from server.agent.services.llm_service import LLMService
from server.agent.segmenter import SentenceSegmenter
from server.pipecat.frames import (
    UserEndHardFrame,
    UserEndSoftFrame,
    AgentInterruptFrame,
    LLMStartFrame,
    LLMDeltaFrame,
    LLMSentenceFrame,
    LLMEndFrame,
)


class LLMProcessor(FrameProcessor):
    """Processes user text through Gemini LLM.

    Design:
    - Receives UserEndHardFrame to trigger LLM generation
    - Can also receive UserEndSoftFrame for early LLM start
    - Streams LLM response through SentenceSegmenter
    - Emits LLMSentenceFrame for each complete sentence
    - Handles AgentInterruptFrame to cancel generation
    """

    def __init__(self, early_start: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._llm = LLMService()
        self._segmenter = SentenceSegmenter()
        self._early_start = early_start
        self._current_task: Optional[asyncio.Task] = None
        self._is_generating = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserEndHardFrame):
            # Trigger LLM generation
            if frame.text and frame.text.strip():
                await self._start_generation(frame.text)
            await self.push_frame(frame)

        elif isinstance(frame, UserEndSoftFrame) and self._early_start:
            # Early LLM start (optional)
            if frame.text and frame.text.strip() and not self._is_generating:
                await self._start_generation(frame.text)
            await self.push_frame(frame)

        elif isinstance(frame, AgentInterruptFrame):
            # Cancel current generation
            await self._cancel_generation()
            await self.push_frame(frame)

        else:
            # Pass through other frames
            await self.push_frame(frame)

    async def _start_generation(self, user_text: str):
        """Start LLM generation task."""
        if self._is_generating:
            return

        self._is_generating = True
        await self.push_frame(LLMStartFrame(user_text=user_text))

        # Run generation in background task
        self._current_task = asyncio.create_task(
            self._generate(user_text)
        )

    async def _generate(self, user_text: str):
        """Generate LLM response and emit sentence frames."""
        full_response = ""
        self._segmenter = SentenceSegmenter()  # Reset segmenter

        try:
            async for delta in self._llm.generate_stream_async(user_text):
                if not self._is_generating:
                    break

                full_response += delta
                await self.push_frame(LLMDeltaFrame(delta=delta))

                # Segment into sentences
                for sentence in self._segmenter.push(delta):
                    await self.push_frame(LLMSentenceFrame(sentence=sentence))

            # Flush remaining text
            if self._is_generating:
                for sentence in self._segmenter.flush_last():
                    await self.push_frame(LLMSentenceFrame(sentence=sentence))

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[LLM] Error: {e}")
        finally:
            self._is_generating = False
            await self.push_frame(LLMEndFrame(full_text=full_response))

    async def _cancel_generation(self):
        """Cancel ongoing LLM generation."""
        self._is_generating = False
        self._llm.request_cancel()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass

    def reset(self):
        """Reset LLM conversation history."""
        self._llm.reset()
        self._is_generating = False
