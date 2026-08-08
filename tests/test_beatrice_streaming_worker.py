"""CI-safe adapter, engine and continuity-aware Worker coverage."""

from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path

import numpy as np

from ai.beatrice.model import BeatriceModelDescriptor, MODEL_API_VERSION
from ai.beatrice.streaming_adapter import BeatriceStreamingAdapter, Float32FIFO
from ai.voice_engine.beatrice import BeatriceVoiceEngine
from ai.voice_worker import VoiceConversionWorker


class FakeModule:
    IN_SAMPLE_RATE = 16_000
    OUT_SAMPLE_RATE = 24_000
    IN_HOP_LENGTH = 160
    OUT_HOP_LENGTH = 240


class FakeConverter:
    def __init__(self) -> None:
        self.config = {}

    def convert(self, audio):
        positions = np.linspace(0, len(audio) - 1, 240)
        return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32), 0

    def set_config(self, **values):
        self.config = dict(values)


def fake_factory():
    return FakeConverter(), FakeModule, {"model_api_version": MODEL_API_VERSION}


class FakeLoader:
    def __init__(self) -> None:
        self.converters: list[FakeConverter] = []

    def create_converter(self, descriptor, config):
        del descriptor
        converter = FakeConverter()
        converter.set_config(**config.to_dict())
        self.converters.append(converter)
        return converter, FakeModule, {
            "model_api_version": MODEL_API_VERSION,
            "num_speakers": 2,
        }


def descriptor() -> BeatriceModelDescriptor:
    return BeatriceModelDescriptor(
        name="test",
        package="test",
        directory=Path("test"),
        metadata_path=Path("test/model.toml"),
        model_name="Test",
        version=MODEL_API_VERSION,
        runtime_requirement=MODEL_API_VERSION,
        speaker_count=2,
        speaker_names=("Alice", "Bob"),
        valid=True,
    )


class BlockingContiguousEngine:
    requires_contiguous_input = True
    is_loaded = True

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.reset_count = 0

    def process_audio(self, audio):
        self.started.set()
        self.release.wait(1.0)
        return np.asarray(audio, dtype=np.float32).copy()

    def reset_stream(self):
        self.reset_count += 1


class BeatriceStreamingWorkerTests(unittest.TestCase):
    def test_fifo_adapter_startup_reset_and_close(self) -> None:
        fifo = Float32FIFO()
        fifo.append(np.arange(5, dtype=np.float32))
        np.testing.assert_array_equal(fifo.pop(3), np.arange(3, dtype=np.float32))
        self.assertEqual(fifo.size, 2)

        adapter = BeatriceStreamingAdapter(fake_factory)
        outputs = [
            adapter.process(np.full(256, 0.1, dtype=np.float32))
            for _ in range(12)
        ]
        self.assertTrue(all(output.shape == (256,) for output in outputs))
        stats = adapter.stats()
        self.assertEqual(stats["startup_silence_samples"], 768)
        self.assertEqual(stats["underflow_count"], 0)
        self.assertEqual(stats["overflow_count"], 0)
        self.assertEqual(stats["input_resampler_drift"], 0.0)
        generation = adapter.converter_generation
        adapter.reset()
        self.assertEqual(adapter.converter_generation, generation + 1)
        self.assertEqual(adapter.stats()["callback_count"], 0)
        adapter.close()
        adapter.close()
        self.assertTrue(adapter.closed)

    def test_unified_engine_load_process_config_reset_unload(self) -> None:
        loader = FakeLoader()
        engine = BeatriceVoiceEngine(descriptor(), loader=loader)
        engine.load_model()
        self.assertTrue(engine.is_loaded)
        for _ in range(4):
            output = engine.process_audio(np.zeros(256, dtype=np.float32))
            self.assertEqual(output.shape, (256,))
        engine.update_config(target_speaker=1, pitch_shift_semitone=2.0)
        self.assertEqual(engine.config.target_speaker, 1)
        generation = engine.adapter.converter_generation
        engine.reset_stream()
        self.assertEqual(engine.adapter.converter_generation, generation + 1)
        self.assertEqual(engine.get_info()["backend"], "beatrice")
        engine.unload_model()
        engine.unload_model()
        self.assertFalse(engine.is_loaded)

    def test_overflow_rejects_new_block_then_resets_off_callback(self) -> None:
        engine = BlockingContiguousEngine()
        worker = VoiceConversionWorker(engine, chunk_size=256, max_queue_size=1)
        self.assertTrue(worker.start())
        block = np.zeros(256, dtype=np.float32)
        try:
            self.assertTrue(worker.put(block))
            self.assertTrue(engine.started.wait(1.0))
            self.assertTrue(worker.put(block))
            self.assertFalse(worker.put(block))
            self.assertEqual(worker.continuity_error_count, 1)
            self.assertTrue(worker.recovery_pending)
            engine.release.set()
            deadline = time.monotonic() + 2.0
            while worker.recovery_count == 0 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(engine.reset_count, 1)
            self.assertEqual(worker.recovery_count, 1)
            self.assertFalse(worker.recovery_pending)
            self.assertIsNone(worker.get_nowait())
            self.assertTrue(worker.put(block))
            self.assertIsNotNone(worker.get(timeout=1.0))
        finally:
            engine.release.set()
            self.assertTrue(worker.stop(timeout=2.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
