from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
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

    def test_quality_gate_and_sequential_parameter_selection_flow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_directory = root / "voice_model_A"
            model_directory.mkdir()
            pth = model_directory / "voice.pth"
            pth.write_bytes(b"fake")
            index_path = model_directory / "voice.index"
            index_path.write_bytes(b"fake-index")
            descriptor = SimpleNamespace(
                name="voice",
                pth_path=pth,
                directory=model_directory,
                index_path=index_path,
                profile=RVCModelProfile(
                    name="voice",
                    voice_dir=model_directory,
                    model_file=pth,
                    inference=RVCInferenceConfig(index_rate=0.60),
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
                self.assertEqual(dialog._model_folder_name(), "voice_model_A")
                self.assertEqual(
                    dialog.profile_name.text(), "voice_model_A - 我的日常配置"
                )
                self.assertEqual(
                    dialog._default_profile_filename(),
                    "voice_model_A_voice_profile.json",
                )
                self.assertEqual(
                    dialog._safe_filename_component('model: A?'),
                    "model_ A_",
                )
                self.assertIn("模型文件夹名", dialog.profile_name.toolTip())
                self.assertIn(
                    "voice_model_A_voice_profile.json",
                    dialog.save_profile_button.toolTip(),
                )
                t = np.arange(sample_rate * 6, dtype=np.float32) / sample_rate
                gate = ((t % 1.0) < 0.75).astype(np.float32)
                dialog._audio = (0.2 * np.sin(2 * np.pi * 170 * t) * gate).astype(
                    np.float32
                )
                dialog._analyze_audio()

                self.assertTrue(dialog._quality.is_acceptable)
                self.assertTrue(dialog.generate_button.isEnabled())
                self.assertEqual(dialog._search.current.stage, "pitch_coarse")
                self.assertTrue(dialog.index_spin.isEnabled())
                self.assertEqual(len(dialog._candidate_audio()), len(dialog._audio))
                self.assertTrue(np.array_equal(dialog._candidate_audio(), dialog._audio))

                evaluation = CandidateEvaluation(90, 88, 92, 91, True)

                def select_candidate(candidate_number: int, filename: str) -> None:
                    index = candidate_number - 1
                    stage = dialog._search.current.stage
                    candidate = CandidateResult(
                        candidate_id=f"candidate-{candidate_number}",
                        label=f"方案 {candidate_number}",
                        parameters=dialog._search.current.candidates[index],
                        audio_path=str(root / filename),
                        inference_ms=10.0,
                        evaluation=evaluation,
                    )
                    dialog._generated_stage = stage
                    dialog._display_results = [candidate]
                    dialog._select_candidate(0)

                select_candidate(4, "coarse.wav")
                self.assertEqual(dialog._search.current.stage, "pitch_fine")
                self.assertTrue(dialog.generate_button.isEnabled())
                self.assertFalse(dialog.apply_button.isEnabled())

                select_candidate(2, "fine.wav")
                self.assertEqual(dialog._search.current.stage, "index_rate")
                self.assertIn("目标音色强度", dialog.generate_button.text())

                select_candidate(2, "index.wav")
                self.assertEqual(dialog._search.current.stage, "protect")
                self.assertIn("辅音清晰度", dialog.generate_button.text())

                select_candidate(2, "protect.wav")
                self.assertEqual(dialog._search.current.stage, "rms_mix_rate")
                self.assertIn("音量动态", dialog.generate_button.text())

                select_candidate(2, "rms.wav")
                self.assertIsNotNone(dialog._search.final_parameters)
                self.assertEqual(
                    [round_.stage for round_ in dialog._search.history],
                    [
                        "pitch_coarse",
                        "pitch_fine",
                        "index_rate",
                        "protect",
                        "rms_mix_rate",
                    ],
                )
                self.assertEqual(dialog.pitch_spin.value(), 0)
                self.assertTrue(dialog.apply_button.isEnabled())
                self.assertTrue(dialog.save_profile_button.isEnabled())
                dialog._inspection = SimpleNamespace(model_hash="test-hash")
                with patch(
                    "gui.customization_dialog.QFileDialog.getSaveFileName",
                    return_value=("", ""),
                ) as chooser:
                    dialog._save_profile()
                default_path = chooser.call_args.args[2]
                self.assertEqual(
                    Path(default_path).name, "voice_model_A_voice_profile.json"
                )
            finally:
                dialog.done(QDialog.DialogCode.Rejected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
