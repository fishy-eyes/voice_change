# 智能音色适配 MVP 开发检查点

记录时间：2026-08-02 01:19:22 +08:00  
分支：`master`  
起始提交：`51ef777`

## 当前结论

已完成第一组可独立验证的基础模块，并在此处主动停止，方便断电后恢复。
没有修改 `audio/stream.py`、`ai/rvc_engine.py`、实时 `RVCWorker`、现有普通效果或用户模型/配置；没有提交，也没有 push。

## 开工前核对结果

- 当前全局采样率为 48000 Hz。
- `RVCEngine` 实际配置名为 `pitch_shift`、`f0_method`、`index_rate`、`rms_mix_rate`、`protect`。
- `pitch_shift` 传入外部 RVC Pipeline 时映射为 `f0_up_key`。
- 外部 Pipeline 已确认：`protect < 0.5` 时启用保护混合，值越小保护越强；`rms_mix_rate` 是输出包络占比，值越低越保留输入包络。
- 现有模型流程是 `RVCModelManager -> RVCRuntime -> RVCEngine/RVCWorker -> AIVoiceEffect`。
- 现有 `AudioRecorder` 只保存双工流参数，录音向导需要使用独立 `sounddevice.InputStream`，不能改实时回调。
- 离线候选生成拟在后台 Qt 线程运行；生成期间暂停 `AudioStream`，复用已加载引擎并在结束后恢复原配置和原流状态，避免两个任务争用同一不可重入引擎。

## 已完成文件

- `customization/__init__.py`：导出定制领域数据类型。
- `customization/schemas.py`：已定义 `RecordingQualityResult`、`VoiceAnalysisResult`、`ModelInspectionResult`、`RVCParameterSet`、`CandidateResult`、`CandidateEvaluation`、`SearchRound`、`CustomizationProfile`。
- `customization/quality_checker.py`：单声道归一化、帧级 RMS、静音、削波、动态范围、基频、基频跳变、背景噪声比例、结构化拒绝原因和质量分。
- `customization/voice_analyzer.py`：生成用户声音基础分析结果。
- `customization/recording_session.py`：固定朗读文本、独立输入流录音、文件导入/重采样、录音保存、按固定文本时间比例切分三个搜索片段、统一临时目录管理。
- `tests/test_customization_quality_checker.py`：正常语音、全静音、过短、严重削波、极低音量、NaN、无有效基频共 7 个测试。
- `tests/test_customization_voice_analyzer.py`：基础声音分析测试。

## 已运行验证

```powershell
E:\Anaconda\envs\voice_change\python.exe -m py_compile customization\__init__.py customization\schemas.py customization\quality_checker.py customization\voice_analyzer.py customization\recording_session.py
E:\Anaconda\envs\voice_change\python.exe -m unittest tests.test_customization_quality_checker tests.test_customization_voice_analyzer -v
```

结果：语法检查通过；8 个单元测试全部通过，耗时约 0.07 秒。

首轮测试曾发现连续元音/纯音没有静音底噪帧时阈值过高，现已通过对自适应阈值增加典型响度上限修复，并已重跑通过。

## 尚未实现

1. `model_inspector.py`：模型 SHA-256、checkpoint 结构和 index 可加载性。
2. `parameter_search.py`：pitch 粗搜/精搜和后续参数搜索框架。
3. `candidate_evaluator.py`：候选损坏、时长、音量、高频和连续性筛选。
4. `candidate_generator.py`：后台离线推理、取消、配置恢复和候选文件写入。
5. `profile_store.py`：JSON 保存、加载、模型哈希/index/版本降级检查。
6. `gui/customization_dialog.py`：导入/录音、分析、A/B/C 试听、选择、微调和保存。
7. `gui/rvc_control_panel.py` 与 `gui/i18n.py` 的最小入口和文案改动。
8. 其余三组要求的单元测试、GUI 无头测试和现有回归测试。

## 明天恢复顺序

1. 先运行上面的 8 个测试确认工作区一致。
2. 实现并测试模型检查。
3. 实现并测试搜索状态机、候选评估、候选生成和配置存储。
4. 最后接入独立 PySide6 对话框，再做语法检查和回归。
5. 真实 RVC 模型候选生成仍单独作为昂贵集成验证，运行前先说明预计时间。

## 当前工作区状态

```text
?? customization/
?? docs/customization_mvp_checkpoint_2026-08-02.md
?? tests/test_customization_quality_checker.py
?? tests/test_customization_voice_analyzer.py
```

这些文件都是本次新增，没有覆盖用户现有配置或模型。当前没有本地提交。
