from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from customization.profile_store import ProfileStore
from customization.schemas import (
    CustomizationProfile,
    ModelInspectionResult,
    RVCParameterSet,
    VoiceAnalysisResult,
)


def make_profile(index_path: str | None = None, *, version: int = 1) -> CustomizationProfile:
    now = datetime.now(timezone.utc).isoformat()
    return CustomizationProfile(
        profile_name="daily",
        model=ModelInspectionResult(
            model_hash="sha256:abc",
            model_path="voice.pth",
            index_path=index_path,
            model_version="v2",
            model_sample_rate=40000,
            uses_f0=True,
            has_index=index_path is not None,
            index_loadable=index_path is not None,
            inspection_time=now,
        ),
        input_device_name="Microphone",
        input_sample_rate=48000,
        voice_analysis=VoiceAnalysisResult(
            duration_seconds=18.0,
            rms_mean=0.05,
            peak=0.4,
            clipping_ratio=0.0,
            voiced_frame_ratio=0.7,
            f0_median=150.0,
            f0_p10=120.0,
            f0_p90=190.0,
            pitch_discontinuity_ratio=0.02,
            dynamic_range_db=18.0,
        ),
        parameters=RVCParameterSet(pitch_shift=8, index_rate=0.6),
        search_summary={"pitch_coarse": 8},
        created_at=now,
        updated_at=now,
        profile_version=version,
    )


class ProfileStoreTests(unittest.TestCase):
    def test_save_and_load_are_consistent_and_versioned(self) -> None:
        store = ProfileStore()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            original = make_profile()
            store.save(original, path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            result = store.load(path)
            self.assertEqual(raw["profile_version"], 1)
            self.assertEqual(result.profile, original)
            self.assertIsNone(result.error)

    def test_hash_mismatch_returns_warning(self) -> None:
        store = ProfileStore()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            store.save(make_profile(), path)
            result = store.load(path, expected_model_hash="sha256:different")
            self.assertIsNotNone(result.profile)
            self.assertTrue(any("哈希" in item for item in result.warnings))

    def test_missing_index_degrades_to_zero(self) -> None:
        store = ProfileStore()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            store.save(make_profile(str(Path(directory) / "missing.index")), path)
            result = store.load(path)
            self.assertEqual(result.profile.parameters.index_rate, 0.0)
            self.assertFalse(result.profile.model.has_index)
            self.assertTrue(any("index" in item for item in result.warnings))

    def test_invalid_json_does_not_crash(self) -> None:
        store = ProfileStore()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text("{not json", encoding="utf-8")
            result = store.load(path)
            self.assertIsNone(result.profile)
            self.assertIsNotNone(result.error)


if __name__ == "__main__":
    unittest.main(verbosity=2)
