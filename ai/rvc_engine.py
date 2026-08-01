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
import threading
from pathlib import Path
from time import perf_counter as _perf_counter
from typing import Any, Mapping

import numpy as np
from loguru import logger

from ai.rvc_index_cache import (
    CachedRVCIndex,
    RVCIndexCacheRegistry,
)
from config.rvc_profiles import (
    RVCInferenceConfig,
    RVCModelProfile,
    load_rvc_profile,
)
from config.settings import SAMPLE_RATE


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
        sample_rate: int = SAMPLE_RATE,
        hubert_path: str | Path | None = None,
        rmvpe_path: str | Path | None = None,
        config: RVCInferenceConfig | Mapping[str, Any] | None = None,
        voice_pth_path: str | Path | None = None,
        index_path: str | Path | None = None,
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
        hubert_path : str | Path | None
            Optional Transformers HuBERT directory for an isolated diagnostic
            run. Defaults to ``models_dir / "hubert"``.
        rmvpe_path : str | Path | None
            Optional RMVPE PyTorch checkpoint for an isolated diagnostic run.
            Defaults to ``models_dir / "rmvpe" / "rmvpe.pt"``.
        config : RVCInferenceConfig | Mapping | None
            Optional validated inference configuration. When supplied, it
            replaces the five legacy inference keyword values.
        voice_pth_path : str | Path | None
            Optional explicit voice checkpoint. Relative paths are resolved
            inside ``voice_dir``. The legacy first-``.pth`` lookup remains the
            default.
        index_path : str | Path | None
            Optional explicit feature index. Relative paths are resolved
            inside ``voice_dir``. The legacy first-``.index`` lookup remains
            the default.
        """
        self._voice_dir: Path = Path(voice_dir)
        self._source_dir: Path = Path(source_dir)
        self._models_dir: Path = Path(models_dir)
        self._hubert_path: Path = (
            Path(hubert_path)
            if hubert_path is not None
            else self._models_dir / "hubert"
        )
        self._rmvpe_path: Path = (
            Path(rmvpe_path)
            if rmvpe_path is not None
            else self._models_dir / "rmvpe" / "rmvpe.pt"
        )
        self._voice_pth_path = self._resolve_voice_path(voice_pth_path)
        self._configured_index_path = self._resolve_voice_path(index_path)
        self._config_lock = threading.RLock()
        self._inference_lock = threading.RLock()
        if config is None:
            initial_config = RVCInferenceConfig(
                pitch_shift=pitch_shift,
                f0_method=f0_method,
                index_rate=index_rate,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
            )
        elif isinstance(config, RVCInferenceConfig):
            initial_config = config
        elif isinstance(config, Mapping):
            initial_config = RVCInferenceConfig.from_mapping(config)
        else:
            raise TypeError("config must be RVCInferenceConfig, a mapping, or None")
        self._apply_runtime_config(initial_config)
        self._sample_rate: int = sample_rate
        self._model_loaded: bool = False

        # RVC internals (populated on load_model)
        self._net_g: object = None
        self._hubert_model: object = None
        self._pipeline: object = None
        self._index_path: str = ""
        self._index_cache: CachedRVCIndex | None = None
        self._index_cache_created: bool = False
        self._last_index_cache_info: dict[str, Any] | None = None
        self._tgt_sr: int = 0
        self._if_f0: int = 1
        self._version: str = "v2"
        self._is_half: bool = False
        self._device: str = "cpu"
        self._sid: int = 0
        self._rvc_path_added: bool = False
        self._last_inference_profile: dict[str, float] | None = None

        logger.debug(
            "RVCEngine created: voice_dir={} pitch_shift={} sr={}",
            self._voice_dir.name, self._pitch_shift, self._sample_rate,
        )

    @classmethod
    def from_profile(
        cls,
        profile: RVCModelProfile | str | Path,
        *,
        source_dir: str | Path,
        models_dir: str | Path,
        sample_rate: int = SAMPLE_RATE,
        hubert_path: str | Path | None = None,
        rmvpe_path: str | Path | None = None,
    ) -> "RVCEngine":
        """Construct an unloaded engine from a model profile object or file."""

        loaded_profile = (
            profile
            if isinstance(profile, RVCModelProfile)
            else load_rvc_profile(profile)
        )
        return cls(
            voice_dir=loaded_profile.resolve_voice_dir(models_dir),
            source_dir=source_dir,
            models_dir=models_dir,
            sample_rate=sample_rate,
            hubert_path=hubert_path,
            rmvpe_path=rmvpe_path,
            config=loaded_profile.inference,
            voice_pth_path=loaded_profile.model_file,
            index_path=loaded_profile.index_file,
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
        self.update_config(pitch_shift=value)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def hubert_path(self) -> Path:
        return self._hubert_path

    @property
    def rmvpe_path(self) -> Path:
        return self._rmvpe_path

    @property
    def f0_method(self) -> str:
        return self._f0_method

    @property
    def index_rate(self) -> float:
        return self._index_rate

    @property
    def rms_mix_rate(self) -> float:
        return self._rms_mix_rate

    @property
    def protect(self) -> float:
        return self._protect

    @property
    def config(self) -> RVCInferenceConfig:
        """Return an immutable snapshot of the active inference settings."""

        with self._config_lock:
            return RVCInferenceConfig(
                pitch_shift=self._pitch_shift,
                f0_method=self._f0_method,
                index_rate=self._index_rate,
                rms_mix_rate=self._rms_mix_rate,
                protect=self._protect,
            )

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_half(self) -> bool:
        return self._is_half

    @property
    def last_inference_profile(self) -> dict[str, float] | None:
        """Return a copy of the most recent successful inference timings.

        The external RVC pipeline exposes three cumulative timers. Their
        boundaries are preserved in the field names instead of presenting the
        combined stages as exact HuBERT/index/generator measurements.
        """

        profile = self._last_inference_profile
        return dict(profile) if profile is not None else None

    @property
    def index_cache_info(self) -> dict[str, Any]:
        """Return cache lifecycle and hit/miss statistics."""

        cached = self._index_cache
        if cached is not None:
            info = cached.info()
            info.update(
                enabled=True,
                released=False,
                engine_acquire_miss=int(self._index_cache_created),
                engine_acquire_hit=int(not self._index_cache_created),
                active_registry_entries=RVCIndexCacheRegistry.active_entries(),
            )
            return info
        if self._last_index_cache_info is not None:
            return dict(self._last_index_cache_info)
        return {"enabled": False, "released": False, "path": self._index_path or None}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _load_index_cache(self) -> None:
        if not self._index_path:
            self._last_index_cache_info = None
            return
        cached, created = RVCIndexCacheRegistry.acquire(self._index_path)
        self._index_cache = cached
        self._index_cache_created = created
        self._last_index_cache_info = None
        info = cached.info()
        logger.info(
            "RVCEngine: index cache {} in {:.1f}ms (vectors={} bytes={})",
            "initialized" if created else "reused",
            info["initialization_ms"],
            info["vectors_shape"],
            info["vectors_bytes"],
        )

    def _release_index_cache(self) -> None:
        cached = self._index_cache
        if cached is None:
            return
        RVCIndexCacheRegistry.release(cached)
        info = cached.info()
        info.update(
            enabled=False,
            released=True,
            engine_acquire_miss=int(self._index_cache_created),
            engine_acquire_hit=int(not self._index_cache_created),
            active_registry_entries=RVCIndexCacheRegistry.active_entries(),
        )
        self._last_index_cache_info = info
        self._index_cache = None
        self._index_cache_created = False
        logger.info(
            "RVCEngine: index cache released (read_hits={} reconstruct_hits={})",
            info["read_hits"],
            info["reconstruct_hits"],
        )

    def _resolve_voice_path(self, value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else self._voice_dir / path

    def _apply_runtime_config(self, config: RVCInferenceConfig) -> None:
        self._pitch_shift = config.pitch_shift
        self._f0_method = config.f0_method
        self._index_rate = config.index_rate
        self._rms_mix_rate = config.rms_mix_rate
        self._protect = config.protect

    def update_config(
        self,
        config: RVCInferenceConfig | Mapping[str, Any] | None = None,
        **changes: Any,
    ) -> RVCInferenceConfig:
        """Atomically replace runtime inference parameters.

        Loaded model weights and the realtime Worker are left untouched. An
        in-flight inference keeps its initial immutable snapshot; subsequent
        calls observe the new configuration.
        """

        if config is None:
            next_config = self.config
        elif isinstance(config, RVCInferenceConfig):
            next_config = config
        elif isinstance(config, Mapping):
            next_config = RVCInferenceConfig.from_mapping(config)
        else:
            raise TypeError("config must be RVCInferenceConfig, a mapping, or None")
        if changes:
            next_config = next_config.updated(**changes)
        if (
            self._model_loaded
            and next_config.f0_method == "rmvpe"
            and not self._rmvpe_path.is_file()
        ):
            raise FileNotFoundError(f"RMVPE checkpoint not found: {self._rmvpe_path}")
        with self._config_lock:
            self._apply_runtime_config(next_config)
        logger.info("RVCEngine: runtime config updated: {}", next_config.to_dict())
        return next_config

    def _ensure_rvc_on_path(self) -> None:
        """Add the RVC source tree to sys.path once."""
        if self._rvc_path_added:
            return
        src = str(self._source_dir)
        if src not in sys.path:
            sys.path.insert(0, src)
        if self._rmvpe_path.name != "rmvpe.pt":
            raise ValueError(
                "The external RVC pipeline requires the RMVPE checkpoint to be "
                f"named 'rmvpe.pt': {self._rmvpe_path}"
            )
        os.environ["rmvpe_root"] = str(self._rmvpe_path.parent)
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
        if not (self._hubert_path / "config.json").is_file():
            raise FileNotFoundError(
                f"Transformers HuBERT config not found: {self._hubert_path / 'config.json'}"
            )
        if not (self._hubert_path / "pytorch_model.bin").is_file():
            raise FileNotFoundError(
                "Transformers HuBERT weights not found: "
                f"{self._hubert_path / 'pytorch_model.bin'}"
            )
        if self._f0_method == "rmvpe" and not self._rmvpe_path.is_file():
            raise FileNotFoundError(f"RMVPE checkpoint not found: {self._rmvpe_path}")

        # Use an explicit profile selection or preserve the legacy first-file lookup.
        if self._voice_pth_path is not None:
            voice_pth = self._voice_pth_path
            if not voice_pth.is_file():
                raise FileNotFoundError(f"RVC voice checkpoint not found: {voice_pth}")
        else:
            pth_files = sorted(self._voice_dir.glob("*.pth"))
            if not pth_files:
                raise FileNotFoundError(f"No .pth files in {self._voice_dir}")
            voice_pth = pth_files[0]

        # Use an explicit profile index or preserve the legacy first-file lookup.
        if self._configured_index_path is not None:
            if not self._configured_index_path.is_file():
                raise FileNotFoundError(
                    f"RVC feature index not found: {self._configured_index_path}"
                )
            self._index_path = str(self._configured_index_path)
        else:
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
            _hm.HUBERT_MODEL_PATH = self._hubert_path
            from infer.hubert import load_hubert_model
            self._hubert_model = load_hubert_model(self._device, self._is_half)
            logger.info("RVCEngine: HuBERT loaded from {}", self._hubert_path)
            if self._f0_method == "rmvpe":
                logger.info("RVCEngine: RMVPE checkpoint: {}", self._rmvpe_path)

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
            self._load_index_cache()

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
        self._release_index_cache()
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

        with self._inference_lock:
            self._release_index_cache()
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
            at the configured project sample rate.

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

        runtime_config = self.config
        self._last_inference_profile = None
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
            with self._inference_lock:
                if not self._model_loaded:
                    raise RuntimeError("RVCEngine: model unloaded during preprocessing")
                audio_opt = self._pipeline.pipeline(
                    model=self._hubert_model,
                    net_g=self._net_g,
                    sid=self._sid,
                    audio=audio_16k,
                    times=times,
                    f0_up_key=runtime_config.pitch_shift,
                    f0_method=runtime_config.f0_method,
                    file_index=self._index_path,
                    index_rate=runtime_config.index_rate,
                    if_f0=self._if_f0,
                    tgt_sr=self._tgt_sr,
                    resample_sr=0,
                    rms_mix_rate=runtime_config.rms_mix_rate,
                    version=self._version,
                    protect=runtime_config.protect,
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
        self._last_inference_profile = {
            "total_ms": _t_total * 1000.0,
            "preprocess_ms": _t_pre * 1000.0,
            "pipeline_ms": _t_pipe * 1000.0,
            "postprocess_ms": _t_post * 1000.0,
            "content_index_prepare_ms": times[0] * 1000.0,
            "f0_ms": times[1] * 1000.0,
            "index_synth_ms": times[2] * 1000.0,
            "pipeline_overhead_ms": (
                _t_pipe - times[0] - times[1] - times[2]
            ) * 1000.0,
        }
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
