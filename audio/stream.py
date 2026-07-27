"""Audio stream manager - duplex loopback with effect chain support"""

from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from loguru import logger

from audio.recorder import AudioRecorder
from audio.player import AudioPlayer
from audio.device_manager import DeviceManager
from config.settings import SAMPLE_RATE, CHANNELS, BLOCKSIZE, DTYPE
from effects.manager import EffectManager

ProcessFunc = Callable[[np.ndarray, int, object, object], np.ndarray]


class AudioStream:
    """Duplex audio stream: mic input -> effect chain -> speaker output.

    Processing priority (first match wins):
        1. If effect_manager has effects -> run the effect chain
        2. If _process_func is set -> call it directly
        3. Otherwise -> passthrough (copy input to output)
    """

    def __init__(
        self,
        recorder: Optional[AudioRecorder] = None,
        player: Optional[AudioPlayer] = None,
        effect_manager: Optional[EffectManager] = None,
    ) -> None:
        self._recorder = recorder or AudioRecorder()
        self._player = player or AudioPlayer()
        self._effect_manager = effect_manager
        self._process_func: Optional[ProcessFunc] = None
        self._stream: Optional[sd.Stream] = None
        self._callback_count: int = 0

    # ------------------------------------------------------------------
    # effect manager
    # ------------------------------------------------------------------
    @property
    def effect_manager(self) -> Optional[EffectManager]:
        return self._effect_manager

    @effect_manager.setter
    def effect_manager(self, manager: Optional[EffectManager]) -> None:
        self._effect_manager = manager
        if manager:
            logger.info("effect manager attached ({} effects)", len(manager))
        else:
            logger.info("effect manager detached")

    # ------------------------------------------------------------------
    # legacy single-function hook (kept for backward compatibility)
    # ------------------------------------------------------------------
    def set_process_func(self, func: Optional[ProcessFunc]) -> None:
        self._process_func = func
        logger.info("process func updated: {}", func.__name__ if func else "passthrough")

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._stream is not None:
            logger.warning("audio stream already running")
            return

        rec_params = self._recorder.get_stream_params()
        ply_params = self._player.get_stream_params()

        self._callback_count = 0

        def _callback(indata, outdata, frames, time_info, status):
            self._callback_count += 1
            if status:
                logger.warning("audio status: {}", status)

            # debug: log input volume periodically
            if self._callback_count % 400 == 1:
                in_rms = float(np.sqrt(np.mean(indata ** 2)))
                logger.debug("input RMS: {:.6f}  (count={})", in_rms, self._callback_count)

            # priority: effect_manager > process_func > passthrough
            if self._effect_manager is not None and not self._effect_manager.is_empty:
                try:
                    processed = self._effect_manager.process(indata, frames, time_info, status)
                except Exception as e:
                    logger.error("effect chain error: {}", e)
                    processed = indata
            elif self._process_func is not None:
                try:
                    processed = self._process_func(indata, frames, time_info, status)
                except Exception as e:
                    logger.error("process func error: {}", e)
                    processed = indata
            else:
                processed = indata

            outdata[:] = processed

            # debug: log output volume periodically
            if self._callback_count % 400 == 1:
                out_rms = float(np.sqrt(np.mean(outdata ** 2)))
                logger.debug("output RMS: {:.6f}", out_rms)

        in_dev = rec_params["device"]
        out_dev = ply_params["device"]
        sr = rec_params["samplerate"]

        try:
            self._stream = sd.Stream(
                samplerate=sr,
                blocksize=rec_params["blocksize"],
                dtype=rec_params["dtype"],
                channels=rec_params["channels"],
                callback=_callback,
                device=(in_dev, out_dev),
                latency=(rec_params["latency"], ply_params["latency"]),
            )
            self._stream.start()
        except sd.PortAudioError as e:
            logger.error("failed to open stream at {}Hz: {}", sr, e)
            fallback_sr = None
            if in_dev is not None:
                fallback_sr = sd.query_devices(in_dev)["default_samplerate"]
            elif out_dev is not None:
                fallback_sr = sd.query_devices(out_dev)["default_samplerate"]
            if fallback_sr and fallback_sr != sr:
                logger.info("retrying with device default samplerate: {}Hz", int(fallback_sr))
                self._stream = sd.Stream(
                    samplerate=int(fallback_sr),
                    blocksize=rec_params["blocksize"],
                    dtype=rec_params["dtype"],
                    channels=rec_params["channels"],
                    callback=_callback,
                    device=(in_dev, out_dev),
                    latency=(rec_params["latency"], ply_params["latency"]),
                )
                self._stream.start()
            else:
                raise

        in_name = DeviceManager.get_device_name(in_dev)
        out_name = DeviceManager.get_device_name(out_dev)
        actual_sr = self._stream.samplerate
        logger.info("audio stream started")
        logger.info("  sample rate: {} Hz", int(actual_sr))
        logger.info("  block size: {}", rec_params["blocksize"])
        logger.info("  input device: {}", in_name)
        logger.info("  output device: {}", out_name)
        if self._effect_manager:
            logger.info("  effects: {}", len(self._effect_manager))

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        logger.info("audio stream stopped")

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.active

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception as e:
            logger.warning("error stopping stream: {}", e)
        finally:
            self._stream = None
        logger.info("audio stream stopped")
