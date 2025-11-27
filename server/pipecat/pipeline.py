"""Pipeline builder for the voice agent.

Constructs the pipecat pipeline with all processors.
"""
from pipecat.pipeline.pipeline import Pipeline

from server.pipecat.processors import (
    VADStateProcessor,
    TurnControlProcessor,
    STTProcessor,
    LLMProcessor,
    TTSProcessor,
    BackchannelProcessor,
)


def create_voice_agent_pipeline() -> tuple[Pipeline, dict]:
    """Create the voice agent pipeline.

    Returns:
        Pipeline instance and dict of processor references for control
    """
    # Create processors
    vad_processor = VADStateProcessor()
    turn_processor = TurnControlProcessor()
    stt_processor = STTProcessor()
    llm_processor = LLMProcessor()
    tts_processor = TTSProcessor()
    backchannel_processor = BackchannelProcessor()

    # Link VAD tracker to turn processor
    turn_processor.set_vad_tracker(vad_processor.tracker)

    # Build pipeline
    # Flow: Input → VAD → Turn → STT → Backchannel → LLM → TTS → Output
    pipeline = Pipeline([
        vad_processor,
        turn_processor,
        stt_processor,
        backchannel_processor,
        llm_processor,
        tts_processor,
    ])

    # Return processor references for external control
    processors = {
        "vad": vad_processor,
        "turn": turn_processor,
        "stt": stt_processor,
        "llm": llm_processor,
        "tts": tts_processor,
        "backchannel": backchannel_processor,
    }

    return pipeline, processors


async def initialize_processors(processors: dict):
    """Initialize all processors that need async setup."""
    await processors["stt"].start()
    await processors["tts"].start()
    await processors["backchannel"].start()


async def cleanup_processors(processors: dict):
    """Cleanup all processors."""
    await processors["stt"].stop()
