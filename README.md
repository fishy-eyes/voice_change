# Voice Changer

Windows 实时 AI 变声器。项目使用 PySide6 提供桌面界面，通过 `sounddevice` 建立实时音频流，并把 RVC（Retrieval-based Voice Conversion）作为异步效果接入现有效果链。

当前默认模型为 `modelF`，应用参数来自模型自己的 `profile.json`，不会在推理代码中写死。

## 当前能力

- 麦克风输入与 VB-CABLE/声卡输出设备选择
- 非阻塞 RVC Worker，避免在 PortAudio 回调线程中直接推理
- RVC 模型发现、选择、加载、切换和卸载
- JSON/TOML 模型 profile 与运行时参数更新
- FAISS index 读取和 vectors reconstruct 缓存
- AI Voice 启用/旁路与模型加载状态显示
- 开发用 RVC Advanced 参数面板
  - Pitch
  - Index Rate
  - Protect
  - RMS Mix Rate
- Gain、Echo、Robot 基础音效
- 短 chunk、实时性、阶段耗时和 index cache benchmark
- 模型加载失败时安全回退到基础效果链

## 实时链路

```text
Microphone
  -> AudioStream
  -> EffectManager
  -> AIVoiceEffect
  -> GainEffect
  -> EchoEffect
  -> RobotEffect
  -> Output device / VB-CABLE
```

`AIVoiceEffect` 只负责缓冲和队列交互。实际推理由 `RVCWorker` 在线程中调用 `RVCEngine`。模型切换时，应用会暂停正在运行的 AudioStream，停止旧 Worker，卸载 Engine 和 index cache，然后加载新模型并恢复音频流。

## 目录结构

```text
voice_change/
├── main.py                         # 应用入口和顶层生命周期
├── ai/
│   ├── rvc_engine.py               # RVC 模型、Pipeline、推理和运行时配置
│   ├── rvc_worker.py               # 异步推理 Worker
│   └── rvc_index_cache.py          # FAISS index/vectors 生命周期缓存
├── audio/                          # 录音、播放、设备和实时流
├── config/
│   ├── settings.py                 # 应用和 RVC 后端路径配置
│   ├── rvc_profiles.py             # profile 数据结构、校验和加载
│   └── rvc_profiles/               # profile 示例
├── core/
│   ├── context.py                  # GUI 共享运行时上下文
│   ├── rvc_lifecycle.py            # Engine/Worker 创建和释放
│   ├── rvc_model_manager.py        # models/rvc 模型扫描
│   └── rvc_runtime.py              # 模型选择、切换、启用和退出清理
├── effects/                        # AIVoiceEffect 与基础 DSP 效果
├── gui/
│   ├── main_window.py              # 主窗口与高级参数面板
│   └── rvc_control_panel.py        # AI 开关、模型选择和加载状态
├── models/rvc/                     # 应用管理的 RVC voice 模型
└── tests/                          # 单元、集成、真实模型和 benchmark 测试
```

## 环境要求

- Windows 10/11
- Python 3.11
- PySide6
- sounddevice / PortAudio
- NumPy / SciPy
- PyTorch（建议使用支持 CUDA 的版本）
- RVC 后端所需的 HuBERT/ContentVec、RMVPE、FAISS 等依赖
- 可选：VB-CABLE，用作游戏或语音软件的虚拟麦克风输入

项目当前使用的 Conda 环境：

```powershell
E:\Anaconda\envs\voice_change\python.exe
```

基础应用依赖列在 `requirements.txt`。真实 RVC 推理还依赖外部 RVC 环境中的 PyTorch、FAISS、Transformers、librosa、soundfile 等组件。

## RVC 后端配置

RVC 源码、HuBERT 和 RMVPE 仍由外部后端目录管理，路径配置位于 `config/settings.py`：

```python
RVC_SOURCE_DIR = r"D:\Project_all\rvc_core_test\rvc_source"
RVC_MODELS_DIR = r"D:\Project_all\rvc_core_test\models"
```

其中 `RVC_MODELS_DIR` 提供 HuBERT/ContentVec 和 RMVPE。应用内 voice 模型独立存放在：

```text
models/rvc/<model-name>/
```

## 模型目录与 profile

当前正式模型结构：

```text
models/rvc/modelF/
├── modelF.pth
├── modelF.index
└── profile.json
```

`RVCModelManager` 会扫描 `models/rvc` 的直接子目录。可用模型必须包含合法的 `profile.json` 以及 profile 指向的 `.pth`；`.index` 可以由 profile 指定。

modelF 当前 profile：

```json
{
  "name": "modelF",
  "voice_dir": ".",
  "model_file": "modelF.pth",
  "index_file": "modelF.index",
  "inference": {
    "pitch_shift": 12,
    "f0_method": "rmvpe",
    "index_rate": 0.3,
    "rms_mix_rate": 0.25,
    "protect": 0.33
  }
}
```

模型二进制由 `.gitignore` 排除，不会提交到 Git；profile 文件会随项目版本管理。

## 启动应用

```powershell
cd D:\Project_all\voice_change
E:\Anaconda\envs\voice_change\python.exe main.py
```

当 `ENABLE_AI_VOICE=True` 时，应用启动期间会扫描模型库并自动加载 `RVC_DEFAULT_MODEL`。当前默认模型为 `modelF`，实时 chunk 为 500ms。

如果默认模型加载失败，GUI 仍会启动，Gain/Echo/Robot 和原始音频链路保持可用。

## GUI 使用

1. 在 `Device Selection` 中选择麦克风输入和输出设备，然后点击 `Apply Devices`。
2. 在 `AI Voice` 中确认模型为 `modelF`。
3. 如果模型尚未加载，点击 `Load Model`。
4. 勾选 `Enable AI Voice`。取消勾选只会旁路 AI，不会重新加载模型。
5. 点击 `Start Audio` 启动实时链路。
6. 使用 `RVC Advanced` 调整 Pitch、Index Rate、Protect 和 RMS Mix Rate。更新会从下一次推理开始生效，不会重载 pth/index、Worker 或 Pipeline。
7. 点击 `Stop Audio` 停止音频；关闭应用时会自动释放 Worker、Pipeline、模型和 index cache。

状态示例：

```text
Loaded: modelF (Enabled)
Loaded: modelF (Bypassed)
Not loaded
```

## 常用配置

主要设置位于 `config/settings.py`：

- `ENABLE_AI_VOICE`：启动时是否加载默认 AI 模型
- `RVC_DEFAULT_MODEL`：默认模型名称
- `RVC_MODEL_LIBRARY_DIR`：应用 voice 模型库
- `RVC_CHUNK_SIZE`：实时 RVC 固定 chunk 样本数
- `RVC_INPUT_QUEUE_SIZE`：Worker 输入队列上限
- `RVC_WARMUP_ENABLED` / `RVC_WARMUP_TIMEOUT`：启动暖机设置
- `INPUT_DEVICE` / `AUTO_SELECT_DEVICES`：输入设备选择
- `ENABLE_GAIN` / `ENABLE_ECHO` / `ENABLE_ROBOT`：基础效果初始状态

模型特有的 Pitch、F0、Index、Protect 和 RMS 参数应写入对应的 profile，而不是修改推理实现。

## 测试

以下命令均使用项目 Conda 环境执行。

基础编译检查：

```powershell
E:\Anaconda\envs\voice_change\python.exe -m py_compile main.py ai\rvc_engine.py core\rvc_runtime.py gui\main_window.py
```

配置、模型管理和 GUI 单元测试：

```powershell
E:\Anaconda\envs\voice_change\python.exe -m unittest -v `
  tests.test_rvc_config `
  tests.test_rvc_model_manager `
  tests.test_rvc_runtime `
  tests.test_rvc_model_gui `
  tests.test_main_rvc_wiring
```

真实模型 lifecycle 与 Worker：

```powershell
E:\Anaconda\envs\voice_change\python.exe -u tests\test_rvc_application_model_flow.py
E:\Anaconda\envs\voice_change\python.exe -u tests\test_rvc_app_lifecycle.py
E:\Anaconda\envs\voice_change\python.exe -u tests\test_rvc_worker.py
E:\Anaconda\envs\voice_change\python.exe -u tests\test_rvc_worker_stream.py
```

缓存、性能与 profiling：

```powershell
E:\Anaconda\envs\voice_change\python.exe -u tests\test_rvc_index_cache.py
E:\Anaconda\envs\voice_change\python.exe -u tests\test_rvc_realtime_benchmark.py
E:\Anaconda\envs\voice_change\python.exe -u tests\test_rvc_inference_profile_benchmark.py
```

真实模型测试需要正确的 RVC 后端路径和可用的模型文件，GPU 第一次推理通常包含明显的冷启动开销。

## 生成文件与 Git

以下内容不会进入版本库：

- `models/**/*.pth`、`models/**/*.index` 等模型二进制
- `tests/output/` benchmark JSON 和试听 WAV
- `tests/assets/` 本地测试音频
- Python 缓存、日志、IDE 配置

提交前建议执行：

```powershell
git status --short
git diff --check
```
