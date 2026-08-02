"""Independent PySide6 wizard for offline RVC parameter customization."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)

from audio.device_manager import DeviceManager
from customization.candidate_generator import CandidateGenerator
from customization.model_inspector import ModelInspector
from customization.parameter_search import ParameterSearch
from customization.profile_store import ProfileStore
from customization.quality_checker import RecordingQualityChecker
from customization.recording_session import (
    RecordingSession,
    STANDARD_RECORDING_TEXT,
    TemporaryAudioDirectory,
)
from customization.schemas import (
    CandidateResult,
    CustomizationProfile,
    RVCParameterSet,
)
from customization.voice_analyzer import VoiceAnalyzer


class CandidateTask(QObject):
    """One background search round; the realtime stream is paused by the GUI."""

    finished = Signal(object, object)
    failed = Signal(str)
    progress = Signal(int, int, object)

    def __init__(
        self,
        *,
        engine,
        effect,
        descriptor,
        audio: np.ndarray,
        parameters: list[RVCParameterSet],
        output_directory: Path,
        sample_rate: int,
        cancellation: threading.Event,
        inspection=None,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.effect = effect
        self.descriptor = descriptor
        self.audio = np.asarray(audio, dtype=np.float32)
        self.parameters = parameters
        self.output_directory = output_directory
        self.sample_rate = sample_rate
        self.cancellation = cancellation
        self.inspection = inspection

    @Slot()
    def run(self) -> None:
        try:
            inspection = self.inspection or ModelInspector().inspect(
                self.descriptor.pth_path,
                self.descriptor.index_path,
            )
            has_valid_index = inspection.has_index and inspection.index_loadable
            parameters = [
                value if has_valid_index else replace(value, index_rate=0.0)
                for value in self.parameters
            ]

            worker = getattr(self.effect, "worker", None)
            if worker is not None:
                worker.clear_queues()
                while worker.is_inferencing and not self.cancellation.is_set():
                    time.sleep(0.01)
                worker.clear_queues()
            if self.cancellation.is_set():
                self.finished.emit(inspection, [])
                return

            generator = CandidateGenerator(
                self.engine,
                self.output_directory,
                sample_rate=self.sample_rate,
            )
            results = generator.generate(
                self.audio,
                parameters,
                cancel_event=self.cancellation,
                progress=lambda current, total, result: self.progress.emit(
                    current, total, result
                ),
            )
            self.finished.emit(inspection, results)
        except Exception as exc:
            logger.exception("定制候选任务失败")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class CustomizationDialog(QDialog):
    """A practical MVP: recording analysis, pitch A/B/C and profile saving."""

    PAGE_SIZE = 3

    def __init__(self, context, descriptor, *, language: str = "zh", parent=None) -> None:
        super().__init__(parent)
        self.context = context
        self.descriptor = descriptor
        self.language = "zh" if language == "zh" else "en"
        runtime = getattr(context, "rvc_runtime", None)
        self.sample_rate = int(getattr(runtime, "sample_rate", 48000))
        self._temporary = TemporaryAudioDirectory()
        self._recording = RecordingSession(
            sample_rate=self.sample_rate,
            device=getattr(context, "input_device", None),
        )
        self._audio: np.ndarray | None = None
        self._quality = None
        self._analysis = None
        self._inspection = None
        self._search: ParameterSearch | None = None
        self._final_parameters: RVCParameterSet | None = None
        self._results: list[CandidateResult] = []
        self._display_results: list[CandidateResult] = []
        self._page = 0
        self._generated_stage: str | None = None
        self._thread: QThread | None = None
        self._task: CandidateTask | None = None
        self._cancellation = threading.Event()
        self._stream_was_running = False
        self._close_after_worker = False

        self.setWindowTitle(
            "智能音色适配 / 定制微调"
            if self.language == "zh"
            else "Smart Timbre Customization"
        )
        self.setMinimumSize(820, 700)
        self.resize(900, 760)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        model_name = getattr(self.descriptor, "name", "RVC")
        self.model_label = QLabel(f"模型 / Model: {model_name}")
        self.model_label.setStyleSheet("font-weight: 600; font-size: 15px;")
        root.addWidget(self.model_label)

        instruction = QLabel(
            "请自然朗读下列固定文本约 15～25 秒；不合格录音不会进入候选生成。"
        )
        instruction.setWordWrap(True)
        root.addWidget(instruction)
        self.script_text = QTextEdit(STANDARD_RECORDING_TEXT)
        self.script_text.setReadOnly(True)
        self.script_text.setMaximumHeight(92)
        root.addWidget(self.script_text)

        audio_row = QHBoxLayout()
        self.import_button = QPushButton("导入 WAV")
        self.import_button.clicked.connect(self._import_audio)
        self.record_button = QPushButton("开始录音")
        self.record_button.clicked.connect(self._start_recording)
        self.stop_record_button = QPushButton("停止录音")
        self.stop_record_button.clicked.connect(self._stop_recording)
        self.stop_record_button.setEnabled(False)
        self.save_recording_button = QPushButton("保存完整录音")
        self.save_recording_button.clicked.connect(self._save_recording)
        self.save_recording_button.setEnabled(False)
        self.analyze_button = QPushButton("检查录音质量")
        self.analyze_button.clicked.connect(self._analyze_audio)
        self.analyze_button.setEnabled(False)
        for button in (
            self.import_button,
            self.record_button,
            self.stop_record_button,
            self.save_recording_button,
            self.analyze_button,
        ):
            audio_row.addWidget(button)
        root.addLayout(audio_row)

        self.quality_label = QLabel("尚未导入或录制语音。")
        self.quality_label.setWordWrap(True)
        self.quality_label.setStyleSheet("padding: 8px; background: #f4f6fa;")
        root.addWidget(self.quality_label)

        search_row = QHBoxLayout()
        self.generate_button = QPushButton("生成 Pitch 粗搜索候选")
        self.generate_button.clicked.connect(self._generate_current_round)
        self.generate_button.setEnabled(False)
        self.cancel_button = QPushButton("取消生成")
        self.cancel_button.clicked.connect(self._cancel_generation)
        self.cancel_button.setEnabled(False)
        search_row.addWidget(self.generate_button)
        search_row.addWidget(self.cancel_button)
        root.addLayout(search_row)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)

        candidate_group = QGroupBox("候选试听（每页最多三个，不显示底层参数）")
        candidate_layout = QGridLayout(candidate_group)
        self._candidate_labels: list[QLabel] = []
        self._candidate_status: list[QLabel] = []
        self._play_buttons: list[QPushButton] = []
        self._select_buttons: list[QPushButton] = []
        for row in range(self.PAGE_SIZE):
            label = QLabel(f"方案 {chr(65 + row)}")
            status = QLabel("等待生成")
            play = QPushButton("播放")
            select = QPushButton("选择")
            play.clicked.connect(lambda checked=False, slot=row: self._play_candidate(slot))
            select.clicked.connect(
                lambda checked=False, slot=row: self._select_candidate(slot)
            )
            play.setEnabled(False)
            select.setEnabled(False)
            candidate_layout.addWidget(label, row, 0)
            candidate_layout.addWidget(status, row, 1)
            candidate_layout.addWidget(play, row, 2)
            candidate_layout.addWidget(select, row, 3)
            self._candidate_labels.append(label)
            self._candidate_status.append(status)
            self._play_buttons.append(play)
            self._select_buttons.append(select)
        navigation = QHBoxLayout()
        self.previous_button = QPushButton("上一页")
        self.next_button = QPushButton("下一页")
        self.previous_button.clicked.connect(lambda: self._change_page(-1))
        self.next_button.clicked.connect(lambda: self._change_page(1))
        navigation.addStretch()
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.next_button)
        candidate_layout.addLayout(navigation, self.PAGE_SIZE, 0, 1, 4)
        root.addWidget(candidate_group)

        manual_group = QGroupBox("推荐参数与手动微调")
        manual_layout = QGridLayout(manual_group)
        self.pitch_spin = QSpinBox()
        self.pitch_spin.setRange(-24, 24)
        self.index_spin = self._rate_spin(0.0, 1.0)
        self.protect_spin = self._rate_spin(0.0, 0.5)
        self.rms_spin = self._rate_spin(0.0, 1.0)
        rows = (
            ("音高", self.pitch_spin),
            ("目标音色强度", self.index_spin),
            ("辅音清晰度", self.protect_spin),
            ("音量稳定度", self.rms_spin),
        )
        for row, (label, control) in enumerate(rows):
            manual_layout.addWidget(QLabel(label), row, 0)
            manual_layout.addWidget(control, row, 1)
        self.profile_name = QLineEdit("我的日常配置")
        manual_layout.addWidget(QLabel("配置名称"), len(rows), 0)
        manual_layout.addWidget(self.profile_name, len(rows), 1)
        profile_buttons = QHBoxLayout()
        self.apply_button = QPushButton("应用到实时变声")
        self.apply_button.clicked.connect(self._apply_parameters)
        self.save_profile_button = QPushButton("保存 JSON 配置")
        self.save_profile_button.clicked.connect(self._save_profile)
        self.load_profile_button = QPushButton("加载 JSON 配置")
        self.load_profile_button.clicked.connect(self._load_profile)
        self.apply_button.setEnabled(False)
        self.save_profile_button.setEnabled(False)
        profile_buttons.addWidget(self.apply_button)
        profile_buttons.addWidget(self.save_profile_button)
        profile_buttons.addWidget(self.load_profile_button)
        manual_layout.addLayout(profile_buttons, len(rows) + 1, 0, 1, 2)
        root.addWidget(manual_group)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.reject)
        close_row.addWidget(close_button)
        root.addLayout(close_row)

    @staticmethod
    def _rate_spin(minimum: float, maximum: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        return spin

    def _runtime(self):
        return getattr(self.context, "rvc_runtime", None)

    def _pause_stream(self) -> None:
        stream = getattr(self.context, "audio_stream", None)
        self._stream_was_running = bool(stream is not None and stream.is_running)
        if self._stream_was_running:
            stream.stop()

    def _resume_stream(self) -> None:
        stream = getattr(self.context, "audio_stream", None)
        if self._stream_was_running and stream is not None and not stream.is_running:
            try:
                stream.start()
            except Exception as exc:
                logger.error("定制流程恢复 AudioStream 失败: {}", exc)
        self._stream_was_running = False

    @Slot()
    def _import_audio(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "导入标准语音",
            "",
            "Wave audio (*.wav);;Audio files (*.wav *.flac *.ogg)",
        )
        if not path:
            return
        try:
            self._audio = RecordingSession.load_file(path, self.sample_rate)
        except Exception as exc:
            QMessageBox.warning(self, "导入失败", str(exc))
            return
        self.analyze_button.setEnabled(True)
        self.save_recording_button.setEnabled(True)
        self.quality_label.setText(f"已导入：{Path(path).name}，等待质量检查。")

    @Slot()
    def _start_recording(self) -> None:
        try:
            self._pause_stream()
            self._recording.start()
        except Exception as exc:
            self._resume_stream()
            QMessageBox.warning(self, "录音失败", str(exc))
            return
        self.record_button.setEnabled(False)
        self.import_button.setEnabled(False)
        self.stop_record_button.setEnabled(True)
        self.quality_label.setText("正在录音，请朗读固定文本……")

    @Slot()
    def _stop_recording(self) -> None:
        try:
            self._audio = self._recording.stop()
        finally:
            self._resume_stream()
        self.record_button.setEnabled(True)
        self.import_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)
        self.analyze_button.setEnabled(bool(self._audio is not None and self._audio.size))
        self.save_recording_button.setEnabled(bool(self._audio is not None and self._audio.size))
        self.quality_label.setText("录音完成，等待质量检查。")

    @Slot()
    def _save_recording(self) -> None:
        if self._audio is None or not self._audio.size:
            return
        path, _selected = QFileDialog.getSaveFileName(
            self, "保存完整录音", "customization_recording.wav", "Wave audio (*.wav)"
        )
        if path:
            RecordingSession.save_file(path, self._audio, self.sample_rate)

    @Slot()
    def _analyze_audio(self) -> None:
        if self._audio is None:
            return
        quality = RecordingQualityChecker().check(self._audio, self.sample_rate)
        self._quality = quality
        reasons = "、".join(quality.rejection_reasons) or "无"
        self.quality_label.setText(
            f"质量分：{quality.quality_score}/100；时长：{quality.duration_seconds:.1f}s；"
            f"有效语音：{quality.effective_voice_seconds:.1f}s；静音：{quality.silence_ratio:.1%}；"
            f"RMS：{quality.rms:.4f}；削波：{quality.clipping_ratio:.2%}；"
            f"基频中位数：{quality.f0_median or 0:.1f}Hz；问题：{reasons}"
        )
        logger.info("录音质量结果: {}", quality)
        if not quality.is_acceptable:
            self.generate_button.setEnabled(False)
            return

        self._analysis = VoiceAnalyzer().analyze(self._audio, self.sample_rate)
        logger.info("用户声音分析结果: {}", self._analysis)
        profile = self.descriptor.profile.inference
        has_index = bool(
            getattr(self.descriptor, "index_path", None)
            and Path(self.descriptor.index_path).is_file()
        )
        base = RVCParameterSet(
            pitch_shift=profile.pitch_shift,
            f0_method=profile.f0_method,
            index_rate=profile.index_rate if has_index else 0.0,
            protect=profile.protect,
            rms_mix_rate=profile.rms_mix_rate,
        )
        self._search = ParameterSearch(has_index=has_index, base=base)
        self.index_spin.setEnabled(has_index)
        if not has_index:
            self.index_spin.setValue(0.0)
        self.generate_button.setText("生成 Pitch 粗搜索候选")
        self.generate_button.setEnabled(True)

    @Slot()
    def _generate_current_round(self) -> None:
        runtime = self._runtime()
        if self._audio is None or self._search is None or runtime is None:
            return
        state = getattr(runtime, "state", None)
        engine = getattr(state, "engine", None)
        effect = getattr(state, "effect", None)
        if engine is None or effect is None or not getattr(engine, "is_loaded", False):
            QMessageBox.warning(self, "模型未加载", "请先在主窗口加载当前 RVC 模型。")
            return

        segments = RecordingSession.split_search_segments(self._audio)
        audio = segments["normal"]
        self._pause_stream()
        self._cancellation = threading.Event()
        self._generated_stage = self._search.current.stage
        output_directory = self._temporary.path / self._generated_stage
        thread = QThread(self)
        task = CandidateTask(
            engine=engine,
            effect=effect,
            descriptor=self.descriptor,
            audio=audio,
            parameters=list(self._search.current.candidates),
            output_directory=output_directory,
            sample_rate=self.sample_rate,
            cancellation=self._cancellation,
            inspection=self._inspection,
        )
        task.moveToThread(thread)
        thread.started.connect(task.run)
        task.progress.connect(self._on_progress)
        task.finished.connect(self._on_candidates_ready)
        task.failed.connect(self._on_generation_failed)
        task.finished.connect(thread.quit)
        task.failed.connect(thread.quit)
        thread.finished.connect(task.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._task = task
        self.progress_bar.setRange(0, len(self._search.current.candidates))
        self.progress_bar.setValue(0)
        self.generate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._clear_candidate_slots("正在后台生成……")
        thread.start()

    @Slot(int, int, object)
    def _on_progress(self, current: int, total: int, result: CandidateResult) -> None:
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.quality_label.setText(
            f"正在生成候选 {current}/{total}；最近一次耗时 {result.inference_ms:.0f} ms"
        )

    @Slot(object, object)
    def _on_candidates_ready(self, inspection, results) -> None:
        self._inspection = inspection
        self._results = list(results)
        valid = [
            result
            for result in self._results
            if result.error is None
            and result.audio_path
            and result.evaluation is not None
            and result.evaluation.is_valid
        ]
        valid.sort(key=lambda item: item.evaluation.technical_quality, reverse=True)
        if len(valid) < 2:
            relaxed = [
                result
                for result in self._results
                if result not in valid
                and result.error is None
                and result.audio_path
                and result.evaluation is not None
                and result.evaluation.technical_quality >= 35
                and result.evaluation.clipping_ratio <= 0.05
            ]
            relaxed.sort(key=lambda item: item.evaluation.technical_quality, reverse=True)
            valid.extend(relaxed)
        self._display_results = valid
        self._page = 0
        self._render_page()
        self.cancel_button.setEnabled(False)
        if len(valid) < 2:
            self.quality_label.setText("合格候选过少，模型可能不适合当前录音；请重新录音或更换模型。")
        else:
            self.quality_label.setText(
                f"候选生成完成：{len(valid)} 个可试听。自动评分仅用于技术筛选，音色偏好由你决定。"
            )

    @Slot(str)
    def _on_generation_failed(self, error: str) -> None:
        self.cancel_button.setEnabled(False)
        self.generate_button.setEnabled(True)
        self.quality_label.setText(f"候选生成失败：{error}")

    @Slot()
    def _on_thread_finished(self) -> None:
        self._resume_stream()
        self._thread = None
        self._task = None
        if self._close_after_worker:
            self._close_after_worker = False
            self.done(QDialog.DialogCode.Rejected)

    @Slot()
    def _cancel_generation(self) -> None:
        self._cancellation.set()
        if self._search is not None:
            self._search.cancel()
        self.cancel_button.setEnabled(False)
        self.quality_label.setText("已请求取消，将在当前候选推理结束后停止。")

    def _clear_candidate_slots(self, message: str) -> None:
        for row in range(self.PAGE_SIZE):
            self._candidate_labels[row].setText(f"方案 {chr(65 + row)}")
            self._candidate_status[row].setText(message)
            self._play_buttons[row].setEnabled(False)
            self._select_buttons[row].setEnabled(False)

    def _render_page(self) -> None:
        start = self._page * self.PAGE_SIZE
        current = self._display_results[start : start + self.PAGE_SIZE]
        self._clear_candidate_slots("本页无候选")
        for row, result in enumerate(current):
            evaluation = result.evaluation
            self._candidate_labels[row].setText(f"方案 {chr(65 + row)}")
            self._candidate_status[row].setText(
                f"技术质量 {evaluation.technical_quality}；稳定程度 {evaluation.stability_score}"
            )
            self._play_buttons[row].setEnabled(bool(result.audio_path))
            self._select_buttons[row].setEnabled(True)
        self.previous_button.setEnabled(self._page > 0)
        self.next_button.setEnabled(start + self.PAGE_SIZE < len(self._display_results))

    def _change_page(self, delta: int) -> None:
        maximum = max(0, (len(self._display_results) - 1) // self.PAGE_SIZE)
        self._page = max(0, min(maximum, self._page + delta))
        self._render_page()

    def _result_for_slot(self, slot: int) -> CandidateResult | None:
        index = self._page * self.PAGE_SIZE + slot
        return self._display_results[index] if index < len(self._display_results) else None

    def _play_candidate(self, slot: int) -> None:
        result = self._result_for_slot(slot)
        if result is None or not result.audio_path:
            return
        try:
            audio, sample_rate = sf.read(result.audio_path, dtype="float32", always_2d=False)
            sd.stop()
            sd.play(audio, sample_rate, blocking=False)
        except Exception as exc:
            QMessageBox.warning(self, "播放失败", str(exc))

    def _select_candidate(self, slot: int) -> None:
        result = self._result_for_slot(slot)
        if result is None or self._search is None:
            return
        candidate_index = int(result.candidate_id.rsplit("-", 1)[-1]) - 1
        if self._generated_stage == "pitch_coarse":
            self._search.choose(candidate_index)
            self.generate_button.setText("生成 Pitch 精搜索候选")
            self.generate_button.setEnabled(True)
            self.quality_label.setText("已选择粗搜索方案。下一步将在其附近生成三个精细候选。")
        elif self._generated_stage == "pitch_fine":
            selected = self._search.current.select(candidate_index)
            self._search.history.append(self._search.current)
            self._final_parameters = selected
            self._populate_manual(selected)
            self.quality_label.setText("Pitch 搜索完成。可手动微调、试听后应用并保存配置。")
            self.apply_button.setEnabled(True)
            self.save_profile_button.setEnabled(True)
            self.generate_button.setEnabled(False)

    def _populate_manual(self, parameters: RVCParameterSet) -> None:
        self.pitch_spin.setValue(parameters.pitch_shift)
        self.index_spin.setValue(parameters.index_rate)
        self.protect_spin.setValue(parameters.protect)
        self.rms_spin.setValue(parameters.rms_mix_rate)
        if self._inspection is not None:
            has_index = self._inspection.has_index and self._inspection.index_loadable
        else:
            has_index = bool(getattr(self.descriptor, "index_path", None))
        self.index_spin.setEnabled(has_index)
        if not has_index:
            self.index_spin.setValue(0.0)

    def _manual_parameters(self) -> RVCParameterSet:
        method = (
            self._final_parameters.f0_method
            if self._final_parameters is not None
            else self.descriptor.profile.inference.f0_method
        )
        return RVCParameterSet(
            pitch_shift=self.pitch_spin.value(),
            f0_method=method,
            index_rate=self.index_spin.value() if self.index_spin.isEnabled() else 0.0,
            protect=self.protect_spin.value(),
            rms_mix_rate=self.rms_spin.value(),
        )

    @Slot()
    def _apply_parameters(self) -> None:
        runtime = self._runtime()
        engine = getattr(getattr(runtime, "state", None), "engine", None)
        if engine is None:
            return
        parameters = self._manual_parameters()
        engine.update_config(**parameters.to_engine_changes())
        self._final_parameters = parameters
        logger.info("最终参数: {}", parameters)
        self.quality_label.setText("参数已应用；实时 Worker 将从下一次推理开始使用。")

    def _make_profile(self) -> CustomizationProfile:
        if self._inspection is None or self._analysis is None:
            raise RuntimeError("请先完成录音分析和候选生成")
        now = datetime.now(timezone.utc).isoformat()
        summary: dict[str, object] = {}
        if self._search is not None:
            for round_ in self._search.history:
                if round_.selected is not None:
                    summary[round_.stage] = round_.selected.to_profile_dict()
        device_name = DeviceManager.get_device_name(getattr(self.context, "input_device", None))
        return CustomizationProfile(
            profile_name=self.profile_name.text().strip() or "我的配置",
            model=self._inspection,
            input_device_name=device_name,
            input_sample_rate=self.sample_rate,
            voice_analysis=self._analysis,
            parameters=self._manual_parameters(),
            search_summary=summary,
            created_at=now,
            updated_at=now,
        )

    @Slot()
    def _save_profile(self) -> None:
        try:
            profile = self._make_profile()
        except Exception as exc:
            QMessageBox.warning(self, "无法保存", str(exc))
            return
        default_directory = Path("config") / "customization_profiles"
        default_directory.mkdir(parents=True, exist_ok=True)
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "保存定制配置",
            str(default_directory / "voice_profile.json"),
            "JSON (*.json)",
        )
        if path:
            ProfileStore().save(profile, path)

    @Slot()
    def _load_profile(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, "加载定制配置", "config/customization_profiles", "JSON (*.json)"
        )
        if not path:
            return
        expected_hash = self._inspection.model_hash if self._inspection is not None else None
        loaded = ProfileStore().load(path, expected_model_hash=expected_hash)
        if loaded.error or loaded.profile is None:
            QMessageBox.warning(self, "配置加载失败", loaded.error or "未知错误")
            return
        self._final_parameters = loaded.profile.parameters
        self._populate_manual(loaded.profile.parameters)
        self.profile_name.setText(loaded.profile.profile_name)
        self.apply_button.setEnabled(True)
        if loaded.warnings:
            QMessageBox.warning(self, "配置警告", "\n".join(loaded.warnings))

    def done(self, result: int) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._close_after_worker = True
            self._cancellation.set()
            self.quality_label.setText("正在等待当前推理结束并安全清理……")
            return
        if self._recording.is_recording:
            self._recording.stop()
        self._resume_stream()
        sd.stop()
        self._temporary.cleanup()
        super().done(result)
