"""Opt-in real-model smoke test for offline customization candidates.

Run explicitly with:

    $env:RUN_RVC_CUSTOMIZATION_INTEGRATION = "1"
    E:\Anaconda\envs\voice_change\python.exe -m unittest \
        tests.test_customization_rvc_integration -v

The test is skipped during ordinary unit-test runs because it loads the real
RVC model and GPU stack.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from ai.rvc_engine import RVCEngine
from config.settings import (
    RVC_DEFAULT_MODEL,
    RVC_MODEL_LIBRARY_DIR,
    RVC_MODELS_DIR,
    RVC_SOURCE_DIR,
    RVC_USER_MODELS_FILE,
    SAMPLE_RATE,
)
from core.rvc_model_manager import RVCModelManager
from customization.candidate_generator import CandidateGenerator
from customization.recording_session import RecordingSession
from customization.schemas import RVCParameterSet


@unittest.skipUnless(
    os.environ.get("RUN_RVC_CUSTOMIZATION_INTEGRATION") == "1",
    "set RUN_RVC_CUSTOMIZATION_INTEGRATION=1 to load the real RVC model",
)
class RealRVCCustomizationTests(unittest.TestCase):
    def test_generates_one_real_offline_candidate(self) -> None:
        manager = RVCModelManager(
            RVC_MODEL_LIBRARY_DIR,
            user_models_path=RVC_USER_MODELS_FILE,
        )
        descriptor = manager.get_model(RVC_DEFAULT_MODEL)
        engine = RVCEngine.from_profile(
            descriptor.profile,
            source_dir=RVC_SOURCE_DIR,
            models_dir=RVC_MODELS_DIR,
            sample_rate=SAMPLE_RATE,
        )
        input_path = Path(__file__).parent / "assets" / "input.wav"
        audio = RecordingSession.load_file(input_path, SAMPLE_RATE)
        audio = audio[: SAMPLE_RATE * 3]
        try:
            engine.load_model()
            with tempfile.TemporaryDirectory() as directory:
                results = CandidateGenerator(
                    engine,
                    directory,
                    sample_rate=SAMPLE_RATE,
                ).generate(audio, [RVCParameterSet(index_rate=0.0)])
                self.assertEqual(len(results), 1)
                self.assertIsNone(results[0].error)
                self.assertTrue(results[0].audio_path)
        finally:
            if engine.is_loaded:
                engine.unload_model()


if __name__ == "__main__":
    unittest.main(verbosity=2)
