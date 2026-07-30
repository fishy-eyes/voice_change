"""RVC voice conversion engine.

Responsible for:
- Model loading / unloading
- GPU initialization
- Inference interface
- Buffer management (future)

This is an AI inference engine, NOT a DSP effect.
It will be wrapped by AIVoiceEffect to integrate into the effect chain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


class RVCEngine:
    """RVC voice conversion inference engine.

    Lifecycle:
        1. __init__() — store parameters, do NOT load model yet
        2. load_model() — load weights, initialize GPU
        3. infer() — run voice conversion on a single block
        4. unload_model() — release GPU memory

    This class is intentionally decoupled from BaseEffect /
    EffectManager so it can be tested and used independently.
    """

    def __init__(
        self,
        model_path: str | Path,
        pitch_shift: int = 0,
        sample_rate: int = 44100,
    ) -> None:
        """Store configuration. Does NOT load the model.

        Parameters
        ----------
        model_path : str | Path
            Path to the RVC model file (.pth).
        pitch_shift : int
            Pitch shift in semitones (0 = no shift).
        sample_rate : int
            Expected audio sample rate in Hz.
        """
        self._model_path: Path = Path(model_path)
        self._pitch_shift: int = pitch_shift
        self._sample_rate: int = sample_rate
        self._model_loaded: bool = False
        self._model: object = None  # placeholder for actual model object

        logger.debug(
            "RVCEngine created: model={} pitch_shift={} sr={}",
            self._model_path.name, self._pitch_shift, self._sample_rate,
        )

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def pitch_shift(self) -> int:
        return self._pitch_shift

    @pitch_shift.setter
    def pitch_shift(self, value: int) -> None:
        self._pitch_shift = int(value)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    # ------------------------------------------------------------------
    # model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load RVC model weights and initialize GPU context.

        Raises
        ------
        FileNotFoundError
            If model_path does not exist.
        RuntimeError
            If model loading fails.
        """
        if self._model_loaded:
            logger.warning("RVCEngine: model already loaded, skipping")
            return

        if not self._model_path.exists():
            raise FileNotFoundError(f"RVC model not found: {self._model_path}")

        logger.info("RVCEngine: loading model from {}", self._model_path)

        # TODO: actual RVC model loading
        # - load .pth weights
        # - initialize PyTorch device (CUDA / CPU)
        # - set self._model

        self._model_loaded = True
        logger.info("RVCEngine: model loaded successfully")

    def unload_model(self) -> None:
        """Release model weights and GPU memory."""
        if not self._model_loaded:
            logger.warning("RVCEngine: model not loaded, skipping unload")
            return

        logger.info("RVCEngine: unloading model")

        # TODO: actual cleanup
        # - del self._model
        # - torch.cuda.empty_cache() if on GPU

        self._model = None
        self._model_loaded = False
        logger.info("RVCEngine: model unloaded")

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def infer(self, audio: np.ndarray) -> np.ndarray:
        """Run voice conversion on an audio block.

        Parameters
        ----------
        audio : np.ndarray
            Input audio, shape (samples,), dtype float32, mono.

        Returns
        -------
        np.ndarray
            Converted audio, same shape and dtype as input.

        Raises
        ------
        RuntimeError
            If model is not loaded.
        """
        if not self._model_loaded:
            raise RuntimeError("RVCEngine: model not loaded, call load_model() first")

        # TODO: actual RVC inference
        # - f0 estimation (crepe / rmvpe / harvest)
        # - model forward pass
        # - return converted audio

        # placeholder: passthrough
        logger.trace("RVCEngine.infer: passthrough (not implemented)")
        return audio.copy()

    def __repr__(self) -> str:
        state = "loaded" if self._model_loaded else "not loaded"
        return (
            f"RVCEngine(model={self._model_path.name!r}, "
            f"pitch={self._pitch_shift}, sr={self._sample_rate}, {state})"
        )
