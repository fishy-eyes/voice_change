from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from customization.model_inspector import ModelInspector


class ModelInspectorTests(unittest.TestCase):
    def test_reads_structure_hash_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "voice.pth"
            model.write_bytes(b"test checkpoint bytes")
            index = root / "voice.index"
            index.write_bytes(b"fake index")
            loaded_indexes: list[Path] = []
            inspector = ModelInspector(
                checkpoint_loader=lambda path: {
                    "config": [1, 2, 40000],
                    "version": "v2",
                    "f0": 1,
                },
                index_loader=lambda path: loaded_indexes.append(path),
            )

            result = inspector.inspect(model, index)

            expected_hash = hashlib.sha256(model.read_bytes()).hexdigest()
            self.assertEqual(result.model_hash, f"sha256:{expected_hash}")
            self.assertEqual(result.model_version, "v2")
            self.assertEqual(result.model_sample_rate, 40000)
            self.assertTrue(result.uses_f0)
            self.assertTrue(result.has_index)
            self.assertTrue(result.index_loadable)
            self.assertEqual(loaded_indexes, [index.resolve()])

    def test_bad_index_degrades_without_blocking_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "voice.pth"
            model.write_bytes(b"checkpoint")
            index = root / "voice.index"
            index.write_bytes(b"bad")
            inspector = ModelInspector(
                checkpoint_loader=lambda path: {"config": [48000]},
                index_loader=lambda path: (_ for _ in ()).throw(ValueError("bad index")),
            )

            result = inspector.inspect(model, index)

            self.assertTrue(result.has_index)
            self.assertFalse(result.index_loadable)
            self.assertIn("index_rate=0", result.warning or "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
