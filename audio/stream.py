"""Audio stream manager - duplex loopback with debug logging"""

from typing import Callable, Optional

import numpy as np
import sounddevice as sd
from loguru import logger

from audio.recorder import AudioRecorder
from audio.player import AudioPlayer
from audio.device_manager import DeviceManager
from config.settings import SAMPLE_RATE, CHANNELS, BLOCKSIZE, DTYPE

ProcessFunc = Callable[[np.ndarray, int, object, object], np.ndarray]


class AudioStream:
    """Duplex audio stream: mic input -> process chain -> speaker output.

    Uses a single sd.Stream(input=True, output=True) for minimum latency.
    Inject processing via set_process_func().
    """

    def __init__(
        self,
        recorder: Optional[AudioRecorder] = None,
        player: Optional[AudioPlayer] = None,
    ) -> None:
        self._recorder = recorder or AudioRecorder()
        self._player = player or AudioPlayer()
        self._process_func: Optional[ProcessFunc] = None
        self._stream: Optional[sd.Stream] = None
        self._callback_count: int = 0

    # ------------------------------------------------------------------
    # processing chain
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

            # log input volume every ~2 seconds (assuming 48kHz, 256 block -> ~187 callbacks/sec)
            if self._callback_count % 400 == 1:
                in_rms = float(np.sqrt(np.mean(indata ** 2)))
                logger.debug("input RMS: {:.6f}  (count={})", in_rms, self._callback_count)

            processed = indata
            if self._process_func is not None:
                try:
                    processed = self._process_func(indata, frames, time_info, status)
                except Exception as e:
                    logger.error("process callback error: {}", e)
                    processed = indata

            outdata[:] = processed

            # log output volume periodically
            if self._callback_count % 400 == 1:
                out_rms = float(np.sqrt(np.mean(outdata ** 2)))
                logger.debug("output RMS: {:.6f}", out_rms)

        in_dev = rec_params["device"]
        out_dev = ply_params["device"]
        sr = rec_params["samplerate"]

        # try to open stream; if sample rate mismatch causes failure, fall back to device default
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
            # try with device default samplerate
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
