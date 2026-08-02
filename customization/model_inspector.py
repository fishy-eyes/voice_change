"""Inspect RVC checkpoint metadata without inferring target voice traits."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from customization.schemas import ModelInspectionResult


CheckpointLoader = Callable[[Path], Mapping[str, Any]]
IndexLoader = Callable[[Path], object]


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("RVC checkpoint root must be a mapping")
    return checkpoint


def _load_index(path: Path) -> object:
    import faiss

    return faiss.read_index(str(path))


class ModelInspector:
    def __init__(
        self,
        *,
        checkpoint_loader: CheckpointLoader | None = None,
        index_loader: IndexLoader | None = None,
    ) -> None:
        self._checkpoint_loader = checkpoint_loader or _load_checkpoint
        self._index_loader = index_loader or _load_index

    def inspect(
        self,
        model_path: str | Path,
        index_path: str | Path | None = None,
    ) -> ModelInspectionResult:
        model = Path(model_path).expanduser().resolve()
        if not model.is_file() or model.suffix.lower() != ".pth":
            raise FileNotFoundError(f"RVC .pth not found: {model}")

        checkpoint = self._checkpoint_loader(model)
        config = checkpoint.get("config")
        if not isinstance(config, (list, tuple)) or not config:
            raise ValueError("RVC checkpoint has no usable config")
        try:
            sample_rate = int(config[-1])
        except (TypeError, ValueError) as exc:
            raise ValueError("RVC checkpoint sample rate is invalid") from exc

        selected_index = (
            Path(index_path).expanduser().resolve() if index_path is not None else None
        )
        has_index = bool(
            selected_index is not None
            and selected_index.is_file()
            and selected_index.suffix.lower() == ".index"
        )
        index_loadable = False
        warning = None
        if has_index and selected_index is not None:
            try:
                self._index_loader(selected_index)
                index_loadable = True
            except Exception as exc:
                warning = f"index 无法加载，将使用 index_rate=0: {exc}"
        elif selected_index is not None:
            warning = "index 不存在或扩展名无效，将使用 index_rate=0"

        result = ModelInspectionResult(
            model_hash=sha256_file(model),
            model_path=str(model),
            index_path=str(selected_index) if selected_index is not None else None,
            model_version=str(checkpoint.get("version", "v1")),
            model_sample_rate=sample_rate,
            uses_f0=bool(int(checkpoint.get("f0", 1))),
            has_index=has_index,
            index_loadable=index_loadable,
            inspection_time=datetime.now(timezone.utc).isoformat(),
            warning=warning,
        )
        logger.info(
            "模型检查结果: version={} sample_rate={} f0={} index={}",
            result.model_version,
            result.model_sample_rate,
            result.uses_f0,
            result.index_loadable,
        )
        return result
