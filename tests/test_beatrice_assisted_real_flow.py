"""Real one-candidate assisted-tuning probe; skips without local assets."""

from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import soundfile as sf

from ai.beatrice.model import BeatriceModelManager
from customization.beatrice import (
    BeatriceCandidateGenerator,
    BeatriceParameterSearch,
    BeatriceParameterSet,
    BeatriceTuningCapabilities,
    analyze_beatrice_voice,
)
from customization.recording_session import RecordingSession
from ai.voice_engine.beatrice import BeatriceVoiceEngine


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_ROOT = PROJECT_ROOT / "experiments" / "beatrice_probe" / "assets" / "runtime"
PACKAGE_ROOT = PROJECT_ROOT / "experiments" / "beatrice_probe" / "assets" / "model" / "jvs"
INPUT_WAV = (
    PROJECT_ROOT
    / "experiments"
    / "beatrice_probe"
    / "assets"
    / "input"
    / "common_voice_ja_38833628_16k.wav"
)
HAS_ASSETS = (
    (RUNTIME_ROOT / "beatrice" / "__init__.py").is_file()
    and PACKAGE_ROOT.is_dir()
    and INPUT_WAV.is_file()
)


@unittest.skipUnless(HAS_ASSETS, "external Beatrice runtime/model/input unavailable")
class RealBeatriceAssistedTuningTests(unittest.TestCase):
    def test_each_assisted_stage_produces_isolated_valid_48k_wav(self) -> None:
        descriptor = BeatriceModelManager(PACKAGE_ROOT.parent).get_model(
            PACKAGE_ROOT.name
        )
        source = RecordingSession.load_file(INPUT_WAV, 48_000)
        analysis = analyze_beatrice_voice(source, 48_000)
        parameters = BeatriceParameterSet(
            target_speaker=0,
            pitch_shift_semitone=0.0,
            min_source_pitch=70.0,
            max_source_pitch=420.0,
            formant_shift=0.0,
            vq_num_neighbors=4,
        )
        probe = BeatriceVoiceEngine(descriptor, runtime_root=RUNTIME_ROOT)
        probe.load_model()
        try:
            capabilities = BeatriceTuningCapabilities.from_runtime(
                probe.get_info()["runtime"]
            )
        finally:
            probe.unload_model()
        search = BeatriceParameterSearch(
            parameters,
            analysis,
            capabilities,
            descriptor,
        )
        report = {
            "f0_p5": analysis.f0_p5,
            "f0_p50": analysis.f0_p50,
            "f0_p95": analysis.f0_p95,
            "stages": [],
        }
        with tempfile.TemporaryDirectory() as temp:
            while search.final_parameters is None:
                round_ = search.current
                selected_index = len(round_.candidates) // 2
                selected = round_.candidates[selected_index]
                results = BeatriceCandidateGenerator(
                    descriptor,
                    RUNTIME_ROOT,
                    Path(temp) / round_.stage,
                ).generate(source, [selected])
                self.assertEqual(len(results), 1)
                result = results[0]
                self.assertIsNone(result.error)
                self.assertIsNotNone(result.evaluation)
                self.assertTrue(result.evaluation.is_valid, result.evaluation)
                self.assertIsNotNone(result.audio_path)
                output, rate = sf.read(result.audio_path, dtype="float32")
                self.assertEqual(rate, 48_000)
                self.assertEqual(output.shape, source.shape)
                self.assertTrue(np.isfinite(output).all())
                report["stages"].append(
                    {
                        "stage": round_.stage,
                        "candidate_count": len(round_.candidates),
                        "parameters": selected.to_engine_changes(),
                        "technical_quality": result.evaluation.technical_quality,
                        "clipping_ratio": result.evaluation.clipping_ratio,
                        "inference_ms": result.inference_ms,
                    }
                )
                search.choose(selected_index)
        print("BEATRICE_ASSISTED_REAL=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
