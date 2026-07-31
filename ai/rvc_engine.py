"""RVC voice conversion engine.

Responsible for:
- Model loading / unloading
- GPU initialization
- Inference interface

Delegates to the RVC source tree in rvc_core_test/rvc_source which is
managed separately and must NOT be copied into this project.
"""

from __future__ import annotations

import os
import sys
import gc
from pathlib import Path
from time import perf_counter as _perf_counter

import numpy as np
from loguru import logger


class RVCEngine:
    """RVC voice conversion inference engine.

    Lifecycle:
        1. __init__()  - store parameters, do NOT load model yet
        2. load_model()  - load weights, initialize GPU
        3. infer()  - run voice conversion on a single block
        4. unload_model()  - release GPU memory

    Decoupled from BaseEffect / EffectManager so it can be tested
    and used independently.
    """

    def __init__(
        self,
        voice_dir: str | Path,
        source_dir: str | Path,
        models_dir: str | Path,
        pitch_shift: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        sample_rate: int = 44100,
    ) -> None:
        """Store configuration.  Does NOT load the model.

        Parameters
        ----------
        voice_dir : str | Path
            Directory containing .pth and .index voice files.
        source_dir : str | Path
            Path to the RVC source tree (rvc_source/).
        models_dir : str | Path
            Path to the RVC models root (contains hubert/, rmvpe/).
        pitch_shift : int
            Pitch shift in semitones (f0_up_key, 0 = no shift).
        f0_method : str
            F0 estimation method: "rmvpe", "pm", or "fcpe".
        index_rate : float
            Index matching rate (0.0 - 1.0).
        rms_mix_rate : float
            RMS envelope mix rate.
        protect : float
            Consonant protection (0.0 - 0.5).
        sample_rate : int
            Expected audio sample rate in Hz (project SR).
        """
        self._voice_dir: Path = Path(voice_dir)
        self._source_dir: Path = Path(source_dir)
        self._models_dir: Path = Path(models_dir)
        self._pitch_shift: int = pitch_shift
        self._f0_method: str = f0_method
        self._index_rate: float = index_rate
        self._rms_mix_rate: float = rms_mix_rate
        self._protect: float = protect
        self._sample_rate: int = sample_rate
        self._model_loaded: bool = False

        # RVC internals (populated on load_model)
        self._net_g: object = None
        self._hubert_model: object = None
        self._pipeline: object = None
        self._index_path: str = ""
        self._tgt_sr: int = 0
        self._if_f0: int = 1
        self._version: str = "v2"
        self._is_half: bool = False
        self._device: str = "cpu"
        self._sid: int = 0
        self._rvc_path_added: bool = False

        logger.debug(
            "RVCEngine created: voice_dir={} pitch_shift={} sr={}",
            self._voice_dir.name, self._pitch_shift, self._sample_rate,
        )

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def voice_dir(self) -> Path:
        return self._voice_dir

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
    # internal helpers
    # ------------------------------------------------------------------

    def _ensure_rvc_on_path(self) -> None:
        """Add the RVC source tree to sys.path once."""
        if self._rvc_path_added:
            return
        src = str(self._source_dir)
        if src not in sys.path:
            sys.path.insert(0, src)
        os.environ["rmvpe_root"] = str(self._models_dir / "rmvpe")
        os.environ["index_root"] = str(self._voice_dir)
        self._rvc_path_added = True
        logger.debug("RVCEngine: RVC source added to sys.path: {}", src)

    # ------------------------------------------------------------------
    # model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load RVC model weights and initialize GPU context.

        Raises
        ------
        FileNotFoundError
            If voice_dir or required model files are missing.
        RuntimeError
            If model loading fails.
        """
        if self._model_loaded:
            logger.warning("RVCEngine: model already loaded, skipping")
            return

        if not self._voice_dir.exists():
            raise FileNotFoundError(f"RVC voice dir not found: {self._voice_dir}")

        # Find first .pth in voice_dir
        pth_files = sorted(self._voice_dir.glob("*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"No .pth files in {self._voice_dir}")
        voice_pth = pth_files[0]

        # Find matching .index file
        index_files = sorted(self._voice_dir.glob("*.index"))
        self._index_path = str(index_files[0]) if index_files else ""

        logger.info("RVCEngine: loading model from {}", voice_pth.name)
        if self._index_path:
            logger.info("RVCEngine: index file: {}", Path(self._index_path).name)

        try:
            import torch

            self._ensure_rvc_on_path()

            # --- Load voice checkpoint & build synthesizer ---
            from infer.module.models import (
                SynthesizerTrnMs768NSFsid,
                SynthesizerTrnMs256NSFsid,
                SynthesizerTrnMs768NSFsid_nono,
                SynthesizerTrnMs256NSFsid_nono,
            )

            cpt = torch.load(
                str(voice_pth), map_location="cpu", weights_only=False,
            )
            self._tgt_sr = cpt["config"][-1]
            self._if_f0 = cpt.get("f0", 1)
            self._version = cpt.get("version", "v1")
            cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]

            self._is_half = torch.cuda.is_available()
            self._device = "cuda:0" if self._is_half else "cpu"

            SYNTH = {
                ("v1", 1): SynthesizerTrnMs256NSFsid,
                ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
                ("v2", 1): SynthesizerTrnMs768NSFsid,
                ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
            }
            Cls = SYNTH.get(
                (self._version, self._if_f0),
                SynthesizerTrnMs256NSFsid,
            )
            self._net_g = Cls(*cpt["config"], is_half=self._is_half)
            del self._net_g.enc_q
            self._net_g.load_state_dict(cpt["weight"], strict=False)
            self._net_g.eval().to(self._device)
            if self._is_half:
                self._net_g = self._net_g.half()
            del cpt
            logger.info(
                "RVCEngine: synthesizer loaded  version={} f0={} tgt_sr={}",
                self._version, self._if_f0, self._tgt_sr,
            )

            # --- Load HuBERT ---
            import infer.hubert as _hm
            _hm.HUBERT_MODEL_PATH = self._models_dir / "hubert"
            from infer.hubert import load_hubert_model
            self._hubert_model = load_hubert_model(self._device, self._is_half)
            logger.info("RVCEngine: HuBERT loaded")

            # --- Create Pipeline ---
            from infer.vc.pipeline import Pipeline

            cfg = type("Cfg", (), {})()
            cfg.device = self._device
            cfg.is_half = self._is_half
            if self._is_half:
                cfg.x_pad, cfg.x_query, cfg.x_center, cfg.x_max = 3, 10, 60, 65
            else:
                cfg.x_pad, cfg.x_query, cfg.x_center, cfg.x_max = 1, 6, 38, 41
            self._pipeline = Pipeline(self._tgt_sr, cfg)
            logger.info("RVCEngine: Pipeline created")

            self._model_loaded = True
            logger.info(
                "RVCEngine: model loaded successfully  (device={}, half={})",
                self._device, self._is_half,
            )

        except Exception as e:
            logger.error("RVCEngine: load_model failed: {}", e)
            self._unload_partial()
            raise RuntimeError(f"RVC model loading failed: {e}") from e

    def _unload_partial(self) -> None:
        """Clean up partially loaded state after a failed load."""
        self._net_g = None
        self._hubert_model = None
        self._pipeline = None
        self._model_loaded = False

    def unload_model(self) -> None:
        """Release model weights and GPU memory."""
        if not self._model_loaded:
            logger.warning("RVCEngine: model not loaded, skipping unload")
            return

        logger.info("RVCEngine: unloading model")

        self._net_g = None
        self._hubert_model = None
        self._pipeline = None
        self._model_loaded = False

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("RVCEngine: model unloaded")

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def infer(self, audio: np.ndarray) -> np.ndarray:
        """Run voice conversion on an audio block.

        Parameters
        ----------
        audio : np.ndarray
            Input audio, shape (samples,), dtype float32, mono,
            at the project sample rate (e.g. 44100 Hz).

        Returns
        -------
        np.ndarray
            Converted audio, same shape, dtype float32,
            at the project sample rate.

        Raises
        ------
        RuntimeError
            If model is not loaded.
        """
        if not self._model_loaded:
            raise RuntimeError("RVCEngine: model not loaded, call load_model() first")

        _t_total = _perf_counter()

        # --- Stage 1: Input preprocessing ---
        _t0 = _perf_counter()
        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        import librosa

        original_len = len(audio)
        hubert_sr = 16000
        if self._sample_rate != hubert_sr:
            audio_16k = librosa.resample(
                audio, orig_sr=self._sample_rate, target_sr=hubert_sr,
            )
        else:
            audio_16k = audio.copy()
        audio_max = np.abs(audio_16k).max() / 0.95
        if audio_max > 1:
            audio_16k /= audio_max
        _t_pre = _perf_counter() - _t0

        # --- Stage 2: RVC Pipeline ---
        _t0 = _perf_counter()
        times = [0.0, 0.0, 0.0]
        try:
            audio_opt = self._pipeline.pipeline(
                model=self._hubert_model,
                net_g=self._net_g,
                sid=self._sid,
                audio=audio_16k,
                times=times,
                f0_up_key=self._pitch_shift,
                f0_method=self._f0_method,
                file_index=self._index_path,
                index_rate=self._index_rate,
                if_f0=self._if_f0,
                tgt_sr=self._tgt_sr,
                resample_sr=0,
                rms_mix_rate=self._rms_mix_rate,
                version=self._version,
                protect=self._protect,
            )
        except Exception as e:
            logger.error("RVCEngine: inference failed: {}", e)
            return audio.copy()
        _t_pipe = _perf_counter() - _t0

        # --- Stage 3: Output postprocessing ---
        _t0 = _perf_counter()
        if audio_opt.dtype == np.int16:
            audio_opt = audio_opt.astype(np.float32) / 32768.0
        elif audio_opt.dtype == np.int32:
            audio_opt = audio_opt.astype(np.float32) / 2147483648.0
        elif audio_opt.dtype != np.float32:
            audio_opt = audio_opt.astype(np.float32)

        if self._tgt_sr != self._sample_rate:
            audio_out = librosa.resample(
                audio_opt, orig_sr=self._tgt_sr, target_sr=self._sample_rate,
            )
        else:
            audio_out = audio_opt

        if len(audio_out) != original_len:
            if len(audio_out) > original_len:
                audio_out = audio_out[:original_len]
            else:
                audio_out = np.pad(
                    audio_out, (0, original_len - len(audio_out)),
                )
        _t_post = _perf_counter() - _t0

        _t_total = _perf_counter() - _t_total
        logger.debug(
            "infer: pre={:.1f}ms pipe={:.1f}ms post={:.1f}ms total={:.1f}ms",
            _t_pre * 1000, _t_pipe * 1000, _t_post * 1000, _t_total * 1000,
        )
        logger.debug(
            "  pipeline breakdown: HuBERT={:.1f}ms F0={:.1f}ms Index+Synth={:.1f}ms",
            times[0] * 1000, times[1] * 1000, times[2] * 1000,
        )

        return audio_out.astype(np.float32)

    def __repr__(self) -> str:
        state = "loaded" if self._model_loaded else "not loaded"
        return (
            f"RVCEngine(voice={self._voice_dir.name!r}, "
            f"pitch={self._pitch_shift}, sr={self._sample_rate}, {state})"
        )
