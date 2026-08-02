from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QDialog
from PySide6.QtWidgets import QApplication

from config.rvc_profiles import RVCInferenceConfig, RVCModelProfile
from customization.schemas import CandidateEvaluation, CandidateResult
from gui.customization_dialog import CustomizationDialog


class CustomizationDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_quality_gate_and_pitch_selection_flow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pth = root / "voice.pth"
            pth.write_bytes(b"fake")
            descriptor = SimpleNamespace(
                name="voice",
                pth_path=pth,
                index_path=None,
                profile=RVCModelProfile(
                    name="voice",
                    voice_dir=root,
                    model_file=pth,
                    inference=RVCInferenceConfig(index_rate=0.0),
                ),
            )
            context = SimpleNamespace(
                input_device=None,
                audio_stream=None,
                rvc_runtime=SimpleNamespace(sample_rate=16000),
            )
            dialog = CustomizationDialog(context, descriptor)
            try:
                sample_rate = 16000
                t = np.arange(sample_rate * 6, dtype=np.float32) / sample_rate
                gate = ((t % 1.0) < 0.75).astype(np.float32)
                dialog._audio = (0.2 * np.sin(2 * np.pi * 170 * t) * gate).astype(
                    np.float32
                )
                dialog._analyze_audio()

                self.assertTrue(dialog._quality.is_acceptable)
                self.assertTrue(dialog.generate_button.isEnabled())
                self.assertEqual(dialog._search.current.stage, "pitch_coarse")
                self.assertFalse(dialog.index_spin.isEnabled())

                evaluation = CandidateEvaluation(90, 88, 92, 91, True)
                coarse = CandidateResult(
                    candidate_id="candidate-4",
                    label="方案 D",
                    parameters=dialog._search.current.candidates[3],
                    audio_path=str(root / "coarse.wav"),
                    inference_ms=10.0,
                    evaluation=evaluation,
                )
                dialog._generated_stage = "pitch_coarse"
                dialog._display_results = [coarse]
                dialog._select_candidate(0)
                self.assertEqual(dialog._search.current.stage, "pitch_fine")
                self.assertTrue(dialog.generate_button.isEnabled())

                fine = CandidateResult(
                    candidate_id="candidate-2",
                    label="方案 B",
                    parameters=dialog._search.current.candidates[1],
                    audio_path=str(root / "fine.wav"),
                    inference_ms=10.0,
                    evaluation=evaluation,
                )
                dialog._generated_stage = "pitch_fine"
                dialog._display_results = [fine]
                dialog._select_candidate(0)
                self.assertEqual(dialog.pitch_spin.value(), 0)
                self.assertTrue(dialog.apply_button.isEnabled())
                self.assertTrue(dialog.save_profile_button.isEnabled())
            finally:
                dialog.done(QDialog.DialogCode.Rejected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
