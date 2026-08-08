"""Compact assisted-tuning dialog for the loaded Beatrice backend."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from customization.beatrice import (
    BeatriceCandidateGenerator,
    BeatriceCandidateResult,
    BeatriceParameterSearch,
    BeatriceParameterSet,
    analyze_beatrice_voice,
)
from customization.quality_checker import RecordingQualityChecker
from customization.recording_session import (
    RecordingSession,
    STANDARD_RECORDING_TEXT,
    TemporaryAudioDirectory,
)
from gui.i18n import tr


class BeatriceCandidateTask(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, object)

    def __init__(
        self,
        *,
        descriptor,
        runtime_root: Path,
        audio: np.ndarray,
        parameters: list[BeatriceParameterSet],
        output_directory: Path,
        cancellation: threading.Event,
        realtime_effect=None,
    ) -> None:
        super().__init__()
        self.descriptor = descriptor
        self.runtime_root = runtime_root
        self.audio = np.asarray(audio, dtype=np.float32)
        self.parameters = parameters
        self.output_directory = output_directory
        self.cancellation = cancellation
        self.realtime_effect = realtime_effect

    @Slot()
    def run(self) -> None:
        try:
            worker = getattr(self.realtime_effect, "worker", None)
            if worker is not None:
                worker.clear_queues()
                while worker.is_inferencing and not self.cancellation.is_set():
                    time.sleep(0.01)
                worker.clear_queues()
            if self.cancellation.is_set():
                self.finished.emit([])
                return
            generator = BeatriceCandidateGenerator(
                self.descriptor,
                self.runtime_root,
                self.output_directory,
            )
            results = generator.generate(
                self.audio,
                self.parameters,
                cancel_event=self.cancellation,
                progress=lambda current, total, result: self.progress.emit(
                    current, total, result
                ),
            )
            self.finished.emit(results)
        except Exception as exc:
            logger.exception("Beatrice assisted candidate generation failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class BeatriceCustomizationDialog(QDialog):
    """Record/analyze, then compare isolated candidates stage by stage."""

    STAGE_NAMES = {
        "pitch_coarse": ("音高粗调", "Pitch coarse"),
        "pitch_fine": ("音高微调", "Pitch fine"),
        "formant": ("共振峰", "Formant"),
        "vq_neighbors": ("VQ 邻居数", "VQ neighbors"),
    }

    def __init__(self, context, descriptor, *, language="zh", parent=None) -> None:
        super().__init__(parent)
        self.context = context
        self.descriptor = descriptor
        self.language = "zh" if language == "zh" else "en"
        self.runtime = getattr(context, "beatrice_runtime", None)
        self.manager = getattr(context, "voice_conversion_manager", None)
        self.sample_rate = 48_000
        self._temporary = TemporaryAudioDirectory()
        self._recording = RecordingSession(
            sample_rate=self.sample_rate,
            device=getattr(context, "input_device", None),
        )
        self._audio: np.ndarray | None = None
        self._quality = None
        self._analysis = None
        self._search: BeatriceParameterSearch | None = None
        self._results: list[BeatriceCandidateResult] = []
        self._thread: QThread | None = None
        self._task: BeatriceCandidateTask | None = None
        self._cancellation = threading.Event()
        self._stream_was_running = False
        self._close_after_worker = False
        self.setWindowTitle(
            "Beatrice 智能辅助调参"
            if self.language == "zh"
            else "Beatrice Assisted Tuning"
        )
        self.setMinimumSize(760, 620)
        self._build_ui()

    def _text(self, zh: str, en: str) -> str:
        return zh if self.language == "zh" else en

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        title = QLabel(
            f"{self._text('模型', 'Model')}: {self.descriptor.name}"
        )
        title.setStyleSheet("font-weight: 600; font-size: 15px;")
        root.addWidget(title)
        instruction = QLabel(
            self._text(
                "录制或导入 15–25 秒自然语音。每轮只改变一组参数；技术检测只淘汰损坏输出，最终音色由你选择。",
                "Record or import 15–25 seconds of natural speech. Each round changes one parameter group; technical checks reject only broken output.",
            )
        )
        instruction.setWordWrap(True)
        root.addWidget(instruction)
        script = QTextEdit(STANDARD_RECORDING_TEXT)
        script.setReadOnly(True)
        script.setMaximumHeight(88)
        root.addWidget(script)

        row = QHBoxLayout()
        self.import_button = QPushButton(self._text("导入 WAV", "Import WAV"))
        self.record_button = QPushButton(self._text("开始录音", "Start recording"))
        self.stop_button = QPushButton(self._text("停止录音", "Stop recording"))
        self.analyze_button = QPushButton(self._text("检查并分析", "Check and analyze"))
        self.stop_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        self.import_button.clicked.connect(self._import_audio)
        self.record_button.clicked.connect(self._start_recording)
        self.stop_button.clicked.connect(self._stop_recording)
        self.analyze_button.clicked.connect(self._analyze_audio)
        for button in (self.import_button, self.record_button, self.stop_button, self.analyze_button):
            row.addWidget(button)
        root.addLayout(row)

        self.analysis_label = QLabel(self._text("尚未分析录音。", "Recording not analyzed."))
        self.analysis_label.setWordWrap(True)
        root.addWidget(self.analysis_label)
        self.stage_label = QLabel()
        self.stage_label.setStyleSheet("font-weight: 600;")
        root.addWidget(self.stage_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        root.addWidget(self.progress)
        controls = QHBoxLayout()
        self.generate_button = QPushButton(self._text("生成本轮候选", "Generate candidates"))
        self.cancel_button = QPushButton(self._text("安全取消", "Cancel safely"))
        self.generate_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.generate_button.clicked.connect(self._generate)
        self.cancel_button.clicked.connect(self._cancel)
        controls.addWidget(self.generate_button)
        controls.addWidget(self.cancel_button)
        root.addLayout(controls)

        grid = QGridLayout()
        self.candidate_labels: list[QLabel] = []
        self.play_buttons: list[QPushButton] = []
        self.select_buttons: list[QPushButton] = []
        for index in range(5):
            label = QLabel("—")
            play = QPushButton(self._text("试听", "Play"))
            select = QPushButton(self._text("选择", "Select"))
            play.setEnabled(False)
            select.setEnabled(False)
            play.clicked.connect(lambda _checked=False, i=index: self._play(i))
            select.clicked.connect(lambda _checked=False, i=index: self._select(i))
            grid.addWidget(label, index, 0)
            grid.addWidget(play, index, 1)
            grid.addWidget(select, index, 2)
            self.candidate_labels.append(label)
            self.play_buttons.append(play)
            self.select_buttons.append(select)
        root.addLayout(grid)

        final_row = QHBoxLayout()
        self.apply_button = QPushButton(self._text("应用当前参数", "Apply parameters"))
        self.save_button = QPushButton(self._text("保存说话人预设", "Save speaker preset"))
        self.apply_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.apply_button.clicked.connect(self._apply)
        self.save_button.clicked.connect(self._save)
        final_row.addWidget(self.apply_button)
        final_row.addWidget(self.save_button)
        root.addLayout(final_row)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

    def _import_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "WAV", "", "WAV (*.wav)")
        if not path:
            return
        try:
            self._audio = RecordingSession.load_file(path, self.sample_rate)
        except Exception as exc:
            QMessageBox.warning(self, "Beatrice", str(exc))
            return
        self.analyze_button.setEnabled(True)
        self.status_label.setText(self._text("已导入录音。", "Recording imported."))

    def _start_recording(self) -> None:
        try:
            self._recording.start()
        except Exception as exc:
            QMessageBox.warning(self, "Beatrice", str(exc))
            return
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def _stop_recording(self) -> None:
        try:
            self._audio = self._recording.stop()
        except Exception as exc:
            QMessageBox.warning(self, "Beatrice", str(exc))
            return
        self.record_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.analyze_button.setEnabled(self._audio.size > 0)

    def _analyze_audio(self) -> None:
        if self._audio is None:
            return
        self._quality = RecordingQualityChecker().check(self._audio, self.sample_rate)
        self._analysis = analyze_beatrice_voice(self._audio, self.sample_rate)
        if not self._quality.is_acceptable:
            reasons = ", ".join(self._quality.rejection_reasons)
            self.analysis_label.setText(self._text(f"录音未通过：{reasons}", f"Recording rejected: {reasons}"))
            self.generate_button.setEnabled(False)
            return
        capabilities = self.runtime.get_tuning_capabilities()
        base = BeatriceParameterSet.from_mapping(self.manager.get_current_parameters())
        self._search = BeatriceParameterSearch(
            base, self._analysis, capabilities, self.descriptor
        )
        self.analysis_label.setText(
            tr(
                self.language,
                "beatrice.detected_pitch_range",
                low=f"{self._analysis.f0_p5:.1f}",
                high=f"{self._analysis.f0_p95:.1f}",
            )
            + "\n"
            + tr(
                self.language,
                "beatrice.median_pitch",
                median=f"{self._analysis.f0_p50:.1f}"
            )
            + f"; RMS: {self._analysis.rms:.4f}; "
            + f"voiced: {self._analysis.voiced_ratio:.1%}\n"
            + tr(
                self.language,
                "beatrice.source_pitch_keep_current",
                low=f"{base.min_source_pitch:.1f}",
                high=f"{base.max_source_pitch:.1f}",
            )
        )
        self._prepare_round()

    def _prepare_round(self) -> None:
        if self._search is None:
            return
        stage = self._search.current.stage
        names = self.STAGE_NAMES[stage]
        self.stage_label.setText(
            f"{self._text('当前阶段', 'Current stage')}: {names[0] if self.language == 'zh' else names[1]}"
        )
        self._results = []
        self.generate_button.setEnabled(True)
        for label, play, select in zip(self.candidate_labels, self.play_buttons, self.select_buttons):
            label.setText("—")
            play.setEnabled(False)
            select.setEnabled(False)

    def _pause_stream(self) -> None:
        stream = getattr(self.context, "audio_stream", None)
        self._stream_was_running = bool(stream is not None and stream.is_running)
        if self._stream_was_running:
            stream.stop()

    def _resume_stream(self) -> None:
        stream = getattr(self.context, "audio_stream", None)
        if self._stream_was_running and stream is not None:
            stream.start()
        self._stream_was_running = False

    def _generate(self) -> None:
        if self._audio is None or self._search is None or self._thread is not None:
            return
        runtime_root = getattr(self.runtime, "runtime_path", None)
        if runtime_root is None:
            QMessageBox.warning(self, "Beatrice", "Runtime is not configured")
            return
        self._pause_stream()
        self._cancellation.clear()
        parameters = list(self._search.current.candidates)
        self.progress.setRange(0, len(parameters))
        self.progress.setValue(0)
        self.generate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        thread = QThread(self)
        task = BeatriceCandidateTask(
            descriptor=self.descriptor,
            runtime_root=Path(runtime_root),
            audio=self._audio,
            parameters=parameters,
            output_directory=self._temporary.path,
            cancellation=self._cancellation,
            realtime_effect=getattr(getattr(self.runtime, "state", None), "effect", None),
        )
        task.moveToThread(thread)
        thread.started.connect(task.run)
        task.progress.connect(self._on_progress)
        task.finished.connect(self._on_ready)
        task.failed.connect(self._on_failed)
        task.finished.connect(thread.quit)
        task.failed.connect(thread.quit)
        thread.finished.connect(self._on_finished)
        self._thread = thread
        self._task = task
        thread.start()

    @Slot(int, int, object)
    def _on_progress(self, current: int, total: int, _result) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(current)

    @Slot(object)
    def _on_ready(self, results) -> None:
        self._results = list(results)
        for index, result in enumerate(self._results[:5]):
            valid = bool(result.audio_path and result.evaluation and result.evaluation.is_valid)
            raw_rejected = bool(result.raw_safety and not result.raw_safety.is_safe)
            if result.error:
                detail = result.error
            elif raw_rejected:
                detail = (
                    f"{result.label} · {self._text('技术异常', 'Unsafe raw output')}: "
                    f"{', '.join(result.raw_safety.rejection_reasons)} · "
                    f"peak {result.raw_safety.peak:.3f} · "
                    f"clip {result.raw_safety.clipping_ratio:.3%}"
                )
            elif result.evaluation is not None:
                detail = (
                    f"{result.label} · {self._parameter_summary(result.parameters)} · "
                    f"{result.inference_ms:.0f} ms · quality {result.evaluation.technical_quality}"
                )
            else:
                detail = result.label
            self.candidate_labels[index].setText(detail)
            self.play_buttons[index].setEnabled(valid)
            self.select_buttons[index].setEnabled(valid)
        if not self._results and self._cancellation.is_set():
            self.status_label.setText(self._text("已安全取消。", "Cancelled safely."))
        elif self._results and not any(result.audio_path for result in self._results):
            if self._search is not None:
                next_round = self._search.skip_unsafe_round()
                self.status_label.setText(
                    self._text(
                        "本阶段候选均因技术异常被淘汰，保留进入阶段前的参数。",
                        "All candidates were unsafe; keeping the parameters from before this stage.",
                    )
                )
                if next_round is None:
                    self.apply_button.setEnabled(True)
                    self.save_button.setEnabled(True)
                    self.generate_button.setEnabled(False)
                else:
                    self._prepare_round()

    def _parameter_summary(self, values: BeatriceParameterSet) -> str:
        stage = self._search.current.stage if self._search is not None else ""
        if stage in ("pitch_coarse", "pitch_fine"):
            return f"Pitch {values.pitch_shift_semitone:+.2f} st"
        if stage == "formant":
            return f"Formant {values.formant_shift:+.2f}"
        if stage == "vq_neighbors":
            return f"VQ {values.vq_num_neighbors}"
        return ""

    @Slot(str)
    def _on_failed(self, error: str) -> None:
        self.status_label.setText(error)

    @Slot()
    def _on_finished(self) -> None:
        thread = self._thread
        self._thread = None
        self._task = None
        self.cancel_button.setEnabled(False)
        self._resume_stream()
        if thread is not None:
            thread.deleteLater()
        if self._close_after_worker:
            self._temporary.cleanup()
            super().done(QDialog.DialogCode.Rejected)

    def _cancel(self) -> None:
        self._cancellation.set()
        self.cancel_button.setEnabled(False)
        self.status_label.setText(
            self._text("将在当前 native convert 完成后停止。", "Stopping after the current native convert.")
        )

    def _play(self, index: int) -> None:
        if not 0 <= index < len(self._results):
            return
        path = self._results[index].audio_path
        if path:
            audio, rate = sf.read(path, dtype="float32")
            sd.stop()
            sd.play(audio, rate)

    def _select(self, index: int) -> None:
        if self._search is None or not 0 <= index < len(self._results):
            return
        result = self._results[index]
        if not result.audio_path:
            return
        candidate_index = self._search.current.candidates.index(result.parameters)
        next_round = self._search.choose(candidate_index)
        if next_round is None:
            self.stage_label.setText(self._text("最终参数已选定。", "Final parameters selected."))
            self.apply_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.generate_button.setEnabled(False)
        else:
            self._prepare_round()

    def _apply(self) -> None:
        if self._search is None or self._search.final_parameters is None:
            return
        try:
            self.manager.update_current_parameters(
                **self._search.final_parameters.to_assisted_changes()
            )
        except Exception as exc:
            self.status_label.setText(str(exc))
            return
        self.status_label.setText(self._text("参数已应用。", "Parameters applied."))

    def _save(self) -> None:
        self._apply()
        self.status_label.setText(
            self._text("说话人预设已保存到本机设置。", "Speaker preset saved to local settings.")
        )

    def done(self, result: int) -> None:
        sd.stop()
        if self._recording.is_recording:
            self._recording.stop()
        if self._thread is not None:
            self._close_after_worker = True
            self._cancel()
            return
        self._resume_stream()
        self._temporary.cleanup()
        super().done(result)


__all__ = ["BeatriceCandidateTask", "BeatriceCustomizationDialog"]
