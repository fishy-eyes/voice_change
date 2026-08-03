# Voice Changer

Windows 实时 AI 变声器。程序使用 PySide6 提供桌面界面，以 `sounddevice` 建立实时音频流，并将 RVC 作为异步效果接入音频处理链。

## 项目结构

```text
voice_change/
├── main.py                          # 程序入口与顶层资源生命周期
├── environment.yml                  # Conda 环境入口（Python 3.11 + pip 依赖）
├── requirements.txt                 # 运行依赖与已验证版本
├── requirements-dev.txt             # 测试依赖
├── requirements-build.txt           # Windows 可执行文件构建依赖
├── ai/
│   ├── rvc_engine.py                # RVC 模型、HuBERT、Pipeline 与推理配置
│   ├── rvc_worker.py                # 独立线程中的异步 RVC 推理
│   └── rvc_index_cache.py           # FAISS index 与向量缓存
├── audio/
│   ├── stream.py                    # 麦克风到主输出设备的双工音频流
│   ├── recorder.py / player.py      # 输入、输出设备参数
│   ├── device_manager.py            # Windows 音频设备发现
│   ├── monitor.py                   # 独立的处理后声音自监听
│   └── output_router.py             # 主输出与自监听的音频分流
├── config/
│   ├── settings.py                  # 音频、设备、RVC 后端和默认模型配置
│   ├── rvc_profiles.py              # profile、参数校验与 JSON/TOML 加载
│   ├── rvc_realtime.py              # 三种实时处理预设
│   └── rvc_profiles/                # profile 示例
├── core/
│   ├── context.py                   # GUI 共享应用上下文
│   ├── rvc_model_manager.py         # 内置及外部 RVC 模型发现与登记
│   ├── rvc_runtime.py               # 模型加载、切换、启用和实时模式更新
│   ├── rvc_lifecycle.py             # Engine、Worker 和缓存生命周期
│   └── device_switching.py          # 运行时安全切换音频设备
├── customization/
│   ├── recording_session.py          # 固定文本录音、WAV 导入与完整录音管理
│   ├── quality_checker.py            # 静音、削波、响度、F0 与噪声质量门
│   ├── voice_analyzer.py             # 用户声音统计分析
│   ├── model_inspector.py            # 模型哈希、版本和 index 可用性检查
│   ├── parameter_search.py           # 分层、单参数搜索状态机
│   ├── candidate_generator.py        # 后台离线 RVC 候选生成与响度匹配
│   ├── candidate_evaluator.py        # 候选技术损坏筛选
│   ├── profile_store.py              # 定制 JSON 配置保存、加载与降级
│   └── schemas.py                    # 定制流程结构化数据类型
├── effects/
│   ├── manager.py                   # 有序效果链
│   ├── ai_voice.py                  # RVC 分块、队列、重叠和拼接适配器
│   ├── gain.py                      # 增益
│   ├── echo.py                      # 回声
│   └── robot.py                     # 机器人音色
├── gui/
│   ├── main_window.py               # 横向双列主窗口与高级参数面板
│   ├── rvc_control_panel.py         # AI 开关、模型导入和实时模式选择
│   ├── customization_dialog.py       # 智能音色适配、完整候选试听和配置保存
│   ├── self_monitor_panel.py        # 自监听设备与音量控制
│   └── i18n.py                      # English/中文界面文本
├── packaging/                       # PyInstaller 配置、构建脚本与发布说明
├── rvc_source/                      # 项目内 MIT RVC 推理源码
├── rvc_models/                      # 本机 HuBERT/RMVPE 权重；不进入 Git
├── models/rvc/                      # 项目内模型 profile；二进制不进入 Git
├── config/customization_profiles/   # 本机定制配置；默认不进入 Git
└── tests/                           # 单元、集成、真实模型和 benchmark 测试
```

## 运行逻辑

```text
Microphone
  -> AudioStream
  -> OutputRoutingEffectManager
  -> AIVoiceEffect
  -> GainEffect
  -> Main output / VB-CABLE
  -> Optional Self Monitor / Headphones
```

正式链路统一使用 48000 Hz、mono、float32。`AudioStream` 的 PortAudio 回调只处理短音频块，不在回调线程内执行耗时的模型推理。`AIVoiceEffect` 将输入累积成 RVC 窗口并提交给 `RVCWorker`；Worker 在独立线程中调用 `RVCEngine`，完成后的结果进入输出缓冲区，再按照音频回调所需长度连续返回。

默认 Balanced 模式使用 500 ms chunk 和 50 ms overlap。相邻推理窗口使用线性 crossfade 拼接，减少 chunk 边界突变。输入和输出队列均有容量上限，推理落后时不会让实时回调无限积压。

启动时 `RVCModelManager` 合并扫描 `models/rvc/` 的内置模型和 `config/user_models.json` 中登记的外部模型。`VoiceConversionManager` 选择 Backend，`RVCRuntime` 创建 Engine/Worker、加载所选模型并挂接 `AIVoiceEffect`；模型切换或程序退出时依次停止 Worker、卸载模型并释放 index cache。模型加载失败时自动旁路 AI，最终 Output Gain 仍位于监听分流之前。

主窗口只保留 Backend/模型、音频、监听、Output Gain 和状态控制。RVC 专属设置窗口更新 Pitch、Index Rate、Protect 和 RMS Mix Rate 时，只替换线程安全的推理配置快照，不会重新加载 `.pth`、`.index`、HuBERT、RMVPE 或重建 Pipeline。实时模式切换只安全更新分块缓冲和 Worker 的 chunk shape，不更换已加载模型。

## 使用的算法

### RVC 声音转换

- **HuBERT/ContentVec 内容编码**：提取输入语音的内容特征，降低对原说话人音色的依赖。
- **RMVPE 基频提取**：估计 F0 曲线，为有音高条件的 RVC 模型提供基频信息。profile 也允许选择后端支持的 `pm` 或 `fcpe`。
- **FAISS 特征检索**：从 `.index` 搜索目标声音训练特征，由 Index Rate 控制检索特征与输入特征的混合比例；没有 index 时可使用无 index 模式。
- **RVC Generator/Synthesizer**：组合内容特征、F0、说话人信息和检索特征，合成目标音色波形。
- **RMS envelope mixing**：以 RMS Mix Rate 控制输出响度包络对输入响度的跟随程度。
- **Protect**：保护清辅音、气息等低能量区域，降低过度转换造成的音素损坏。

### 智能音色适配

智能适配复用已经加载的 `RVCEngine` 做离线候选推理，但不改变实时音频链。生成候选时程序会暂停实时 `AudioStream`、等待 Worker 结束当前推理并清空队列；任务结束后恢复原始引擎参数和原音频流状态。

搜索严格逐项进行，用户每次选择后锁定已有结果，再进入下一项，不执行参数笛卡尔积：

1. **Pitch 粗搜索**：比较 `-12/-8/-4/0/+4/+8/+12`。
2. **Pitch 精搜索**：围绕粗搜索结果比较 `-2/0/+2` 半音偏移，最终范围限制在 `-24～+24`。
3. **Index Rate**：有可加载 `.index` 时比较 `0.35/0.60/0.80`；没有有效 index 时自动跳过并固定为 `0`。
4. **Protect**：比较辅音保护强、平衡、目标音色优先三档。当前 RVC 实现中数值越小，原始清辅音特征保护越强。
5. **RMS Mix Rate**：比较保留原声动态、平衡、输出更稳定三档。

每个候选都会转换用户本次朗读的**完整录音**。为降低“更响听起来更好”的试听偏见，写入候选 WAV 前只匹配整体 RMS：

- 各候选具有接近的总体响度；
- 句内强弱、呼吸和包络差异仍然保留，可继续判断 RMS Mix Rate；
- 自动技术评分不因候选更响而加分，音量只用于拒绝静音、极端增益和严重削波；
- 自动评分只排除技术损坏，最终音色偏好由用户试听决定。

录音进入搜索前会检查时长、有效语音、静音比例、RMS、削波、动态范围、F0 有效性、F0 跳变和背景噪声。配置保存时记录模型 SHA-256、index 路径、输入设备、声音分析结果、逐轮选择与最终参数；加载时发现模型哈希变化或 index 丢失会给出警告并安全降级。

### 实时连续性与性能

- **异步生产者/消费者队列**：音频回调生产窗口，Worker 消费并推理，避免 GPU 推理阻塞 PortAudio。
- **固定窗口与 overlap**：固定模型输入 shape，并为相邻窗口保留上下文。
- **线性 crossfade**：对 overlap 使用互补淡入淡出权重，减少拼接处跳变和爆音。
- **FAISS index cache**：模型加载时读取 index 并 reconstruct vectors，后续推理复用缓存，卸载时释放。
- **FP16 CUDA 推理**：检测到 CUDA 时使用 GPU 和 half precision；没有 CUDA 时回退到 CPU/FP32。

### 基础声音效果

- **Gain**：线性缩放采样幅度并限制到安全范围。
- **Echo**：将延迟后的历史采样按衰减系数叠加到当前信号。
- **Robot**：使用固定频率调制输入信号，产生机器人音色。
- **Self Monitor**：把完整效果链的输出复制到独立监听队列；监听异常不会影响 VB-CABLE 主输出。

## 环境配置与使用方法

### 1. 创建环境并安装依赖

已验证环境为 Windows 10/11 x64、Python 3.11.15、PyTorch 2.1.2 + CUDA 12.1。实时 RVC 推荐 NVIDIA GPU；CPU 可以启动基础功能，但模型推理通常无法满足实时要求。

环境文件用途：

- `environment.yml`：创建名为 `voice_change` 的 Conda 环境，并自动安装运行依赖。
- `requirements.txt`：运行 GUI、音频链和 RVC 推理所需的固定版本。
- `requirements-dev.txt`：在运行依赖基础上增加 pytest，用于完整测试套件。
- `requirements-build.txt`：在运行依赖基础上增加固定版本的 PyInstaller，用于生成 Windows Release。

推荐使用 Conda：

```powershell
# 在克隆后的 voice_change 项目根目录执行
conda env create -f environment.yml
conda activate voice_change
python -m pip check
```

如果环境已经存在，可直接使用 pip 同步运行依赖：

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

需要开发和运行全部测试时，再安装：

```powershell
python -m pip install -r requirements-dev.txt
```

`requirements.txt` 默认从 PyTorch 官方 `cu121` 索引安装 CUDA wheel。CPU-only 环境应改装 PyTorch 官方 CPU wheel；不要同时保留 `+cu121` 固定版本。

### 2. 配置 RVC 后端

项目默认直接使用以下本地目录，无需配置绝对路径：

```text
rvc_source/                      # RVC 推理源码
rvc_models/
├── hubert/
│   ├── config.json
│   └── pytorch_model.bin
└── rmvpe/
    └── rmvpe.pt
```

如需临时使用其他位置，可设置 `VOICE_CHANGE_RVC_SOURCE_DIR` 和
`VOICE_CHANGE_RVC_MODELS_DIR` 环境变量覆盖默认目录。

### 3. 准备声音模型

- 在 GUI 中点击 **Import RVC Model**，选择含 `.pth`、可选 `.index` 和可选 `profile.json` 的目录。程序只记录路径，不复制模型。
- 或把模型放入 `models/rvc/<model-name>/`，并提供指向 `.pth` 和可选 `.index` 的 `profile.json`。

没有 profile 的外部模型会使用 RMVPE、Pitch 0、Index Rate 0.30、Protect 0.33、RMS Mix Rate 0.25。存在多个 `.pth` 或 `.index` 时，界面会要求明确选择。

### 4. 启动程序

确认 `config/settings.py` 中启用了 AI 并设置了默认模型：

```python
ENABLE_AI_VOICE = True
RVC_DEFAULT_MODEL = "modelF"
```

运行：

```powershell
python main.py
```

### 5. 在界面中启用变声

1. 使用右上角按钮切换 English/中文。
2. 在 **Device Selection** 中选择麦克风和主输出设备；游戏或语音软件通常使用 VB-CABLE。
3. 在 **AI Voice** 中选择模型并点击 **Load Model**。
4. 勾选 **Enable AI Voice**。
5. 选择 Realtime Mode；默认推荐 **Balanced (500 ms / 50 ms)**。
6. 点击 **Start Audio**。
7. 如需从耳机听到处理后的声音，在 **Self Monitor** 中选择耳机、设置音量并启用。
8. 使用 RVC Advanced 面板实时调节参数；滑条说明会显示向左或向右的声音变化。

### 6. 使用智能音色适配

1. 在 **AI Voice** 中加载目标 RVC 模型。
2. 点击 **智能音色适配 / 定制微调**。
3. 自然朗读界面固定文本约 15～25 秒，或导入一段 WAV。
4. 点击 **检查录音质量**；未通过质量门时根据界面原因重新录制。
5. 按按钮顺序生成并试听 Pitch 粗搜索、Pitch 精搜索、Index Rate、Protect、RMS Mix Rate 候选。
6. 每轮只判断当前提示的听感重点并选择一个方案；播放按钮会播放完整朗读内容。
7. 全部轮次完成后可在推荐参数区域微调，然后点击 **应用到实时变声**。
8. 输入配置名称并点击 **保存 JSON 配置**。

默认配置名称和文件名包含模型文件夹名。例如模型位于 `models/rvc/modelF/`：

```text
配置名称：modelF - 我的日常配置
默认文件：config/customization_profiles/modelF_voice_profile.json
```

保存按钮提示会显示默认文件名。Windows 非法文件名字符会自动替换；配置仍可在保存对话框中重命名或选择其他目录。

### 7. 构建 Windows Release

安装构建依赖后运行脚本；默认使用项目内的 `rvc_source/`：

```powershell
python -m pip install -r requirements-build.txt
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 `
  -PythonExecutable python
```

如需使用其他兼容源码，可额外传入 `-RvcSourceDir path\to\rvc_source`。

构建结果位于 `dist/`：

```text
dist/
├── VoiceChanger-v1.0.0/                  # 解压后的 onedir 应用
├── voice_change-v1.0.0-windows-x64.tar.xz  # GitHub Release 压缩包
└── voice_change-v1.0.0-windows-x64.tar.xz.sha256
```

公开发布包使用 Windows 自带 `tar -xJf voice_change-v1.0.0-windows-x64.tar.xz` 解压，包含应用运行时和 MIT 许可的 RVC 推理源码，但不重新分发 HuBERT、RMVPE、音色 `.pth` 或 `.index` 权重。用户应按照压缩包中的 `README.txt` 放置后端权重，并通过 GUI 导入音色模型。构建出的 EXE 未进行代码签名，Windows SmartScreen 可能显示提示。

## 功能介绍

- Windows 麦克风、扬声器、耳机和 VB-CABLE 设备发现、选择与运行时切换。
- RVC 模型自动发现、外部文件夹导入、持久化登记、加载、切换和安全卸载。
- `.pth` 必选、`.index` 可选；支持 JSON/TOML profile 和无 profile 默认参数。
- Low Latency 325/50、Balanced 500/50、High Quality 500/100 三种实时模式。
- AI Voice 启用/旁路；旁路不会卸载模型，重新启用无需重复加载。
- Pitch、Index Rate、Protect、RMS Mix Rate 运行时高级调节。
- Gain、Echo、Robot 基础效果与 Gain 数值调节。
- 处理后声音自监听，可独立选择耳机设备和监听音量。
- 横向双列 GUI，支持 English/中文即时切换，技术参数名保持英文。
- 模型状态、音频流状态和处理耗时显示。
- 模型 warmup、FAISS index 缓存、有限队列和有序资源清理。
- 录音质量门、模型/index 检查及五阶段单参数智能适配。
- 完整录音候选试听、响度公平处理和按模型文件夹命名的 JSON 配置。
