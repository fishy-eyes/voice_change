"""Process-local, read-only FAISS index cache for RVC model lifecycles."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _path_key(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


class CachedRVCIndex:
    """FAISS-compatible proxy backed by pre-reconstructed vectors."""

    def __init__(
        self,
        path: str | Path,
        index: Any,
        vectors: np.ndarray,
        initialization_ms: float,
    ) -> None:
        self.path = Path(path).resolve()
        self.key = _path_key(self.path)
        self._index = index
        self._vectors = np.asarray(vectors)
        self._vectors.setflags(write=False)
        self._initialization_ms = float(initialization_ms)
        self._lock = threading.RLock()
        self._references = 1
        self._acquire_hits = 0
        self._read_hits = 0
        self._reconstruct_hits = 0
        self._fallback_reconstructs = 0
        self._search_calls = 0

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    def acquire_reference(self) -> None:
        with self._lock:
            self._references += 1
            self._acquire_hits += 1

    def release_reference(self) -> int:
        with self._lock:
            self._references = max(0, self._references - 1)
            return self._references

    def record_read_hit(self) -> None:
        with self._lock:
            self._read_hits += 1

    def search(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            self._search_calls += 1
            return self._index.search(*args, **kwargs)

    def reconstruct_n(self, i0: int = 0, ni: int = -1) -> np.ndarray:
        requested = self.ntotal if ni == -1 else int(ni)
        if int(i0) == 0 and requested == self.ntotal:
            with self._lock:
                self._reconstruct_hits += 1
                return self._vectors
        with self._lock:
            self._fallback_reconstructs += 1
            return self._index.reconstruct_n(i0, ni)

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "path": str(self.path),
                "ntotal": self.ntotal,
                "vectors_shape": list(self._vectors.shape),
                "vectors_dtype": str(self._vectors.dtype),
                "vectors_bytes": int(self._vectors.nbytes),
                "initialization_ms": self._initialization_ms,
                "initialization_count": 1,
                "cache_misses": 1,
                "acquire_hits": self._acquire_hits,
                "read_hits": self._read_hits,
                "reconstruct_hits": self._reconstruct_hits,
                "fallback_reconstructs": self._fallback_reconstructs,
                "search_calls": self._search_calls,
                "references": self._references,
            }

    def __getattr__(self, name: str) -> Any:
        return getattr(self._index, name)


class RVCIndexCacheRegistry:
    """Thread-safe dispatch from external ``faiss.read_index`` to caches."""

    _lock = threading.RLock()
    _entries: dict[str, CachedRVCIndex] = {}
    _faiss_module: Any = None
    _original_read_index: Callable[..., Any] | None = None

    @classmethod
    def _install_dispatcher(cls) -> None:
        if cls._original_read_index is not None:
            return
        import faiss

        cls._faiss_module = faiss
        cls._original_read_index = faiss.read_index
        faiss.read_index = cls._dispatch_read_index

    @classmethod
    def _dispatch_read_index(cls, path: str | os.PathLike[str], *args: Any, **kwargs: Any) -> Any:
        key = _path_key(path)
        with cls._lock:
            cached = cls._entries.get(key)
            original = cls._original_read_index
            if cached is not None:
                cached.record_read_hit()
                return cached
        if original is None:
            raise RuntimeError("RVC index cache dispatcher is not initialized")
        return original(os.fspath(path), *args, **kwargs)

    @classmethod
    def acquire(cls, path: str | Path) -> tuple[CachedRVCIndex, bool]:
        """Acquire a cache reference, returning ``(cache, newly_created)``."""

        resolved = Path(path).resolve()
        key = _path_key(resolved)
        with cls._lock:
            cls._install_dispatcher()
            existing = cls._entries.get(key)
            if existing is not None:
                existing.acquire_reference()
                return existing, False
            if cls._original_read_index is None:
                raise RuntimeError("FAISS read_index is unavailable")
            started_at = time.perf_counter()
            index = cls._original_read_index(str(resolved))
            vectors = index.reconstruct_n(0, index.ntotal)
            initialization_ms = (time.perf_counter() - started_at) * 1000.0
            cached = CachedRVCIndex(
                resolved,
                index,
                vectors,
                initialization_ms,
            )
            cls._entries[key] = cached
            return cached, True

    @classmethod
    def release(cls, cached: CachedRVCIndex) -> None:
        with cls._lock:
            current = cls._entries.get(cached.key)
            if current is not cached:
                return
            if cached.release_reference() == 0:
                del cls._entries[cached.key]

    @classmethod
    def active_entries(cls) -> int:
        with cls._lock:
            return len(cls._entries)
