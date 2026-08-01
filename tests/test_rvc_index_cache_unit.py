"""Fast unit tests for the process-local RVC index cache."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.rvc_index_cache import CachedRVCIndex, RVCIndexCacheRegistry


class _FakeIndex:
    def __init__(self, marker: float) -> None:
        self.ntotal = 4
        self.marker = marker
        self.reconstruct_calls = 0
        self.search_calls = 0

    def reconstruct_n(self, i0: int, ni: int) -> np.ndarray:
        self.reconstruct_calls += 1
        return np.full((ni, 3), self.marker, dtype=np.float32)

    def search(self, values: np.ndarray, k: int):
        self.search_calls += 1
        rows = values.shape[0]
        return (
            np.ones((rows, k), dtype=np.float32),
            np.zeros((rows, k), dtype=np.int64),
        )


class RVCIndexCacheUnitTests(unittest.TestCase):
    def test_proxy_reuses_full_reconstruction_and_tracks_search(self) -> None:
        index = _FakeIndex(1.0)
        vectors = index.reconstruct_n(0, index.ntotal)
        cached = CachedRVCIndex("modelA.index", index, vectors, 12.5)

        first = cached.reconstruct_n(0, cached.ntotal)
        second = cached.reconstruct_n(0, -1)
        cached.search(np.ones((2, 3), dtype=np.float32), 2)

        self.assertIs(first, vectors)
        self.assertIs(second, vectors)
        self.assertFalse(first.flags.writeable)
        self.assertEqual(index.reconstruct_calls, 1)
        self.assertEqual(index.search_calls, 1)
        info = cached.info()
        self.assertEqual(info["reconstruct_hits"], 2)
        self.assertEqual(info["search_calls"], 1)
        self.assertEqual(info["cache_misses"], 1)

    def test_registry_reuses_same_model_then_releases_before_model_switch(self) -> None:
        registry = RVCIndexCacheRegistry
        saved = (
            registry._entries,
            registry._faiss_module,
            registry._original_read_index,
        )
        fake_faiss = types.ModuleType("faiss")
        read_paths: list[str] = []

        def read_index(path: str) -> _FakeIndex:
            read_paths.append(path)
            marker = 1.0 if "modelA" in path else 2.0
            return _FakeIndex(marker)

        fake_faiss.read_index = read_index
        try:
            registry._entries = {}
            registry._faiss_module = None
            registry._original_read_index = None
            with patch.dict(sys.modules, {"faiss": fake_faiss}):
                model_a, created_a = registry.acquire("modelA.index")
                model_a_again, created_again = registry.acquire("modelA.index")
                dispatched = fake_faiss.read_index("modelA.index")

                self.assertTrue(created_a)
                self.assertFalse(created_again)
                self.assertIs(model_a, model_a_again)
                self.assertIs(dispatched, model_a)
                self.assertEqual(len(read_paths), 1)
                self.assertEqual(model_a.info()["acquire_hits"], 1)
                self.assertEqual(model_a.info()["read_hits"], 1)

                registry.release(model_a_again)
                self.assertEqual(registry.active_entries(), 1)
                registry.release(model_a)
                self.assertEqual(registry.active_entries(), 0)

                model_b, created_b = registry.acquire("modelB.index")
                self.assertTrue(created_b)
                self.assertIsNot(model_b, model_a)
                self.assertEqual(len(read_paths), 2)
                np.testing.assert_array_equal(
                    model_b.reconstruct_n(0, -1),
                    np.full((4, 3), 2.0, dtype=np.float32),
                )
                registry.release(model_b)
                self.assertEqual(registry.active_entries(), 0)
        finally:
            if registry._faiss_module is fake_faiss:
                fake_faiss.read_index = read_index
            (
                registry._entries,
                registry._faiss_module,
                registry._original_read_index,
            ) = saved


if __name__ == "__main__":
    unittest.main()
