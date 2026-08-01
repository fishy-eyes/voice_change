# Voice Changer

Windows 实时 AI 变声器。程序使用 PySide6 提供桌面界面，以 `sounddevice` 建立实时音频流，并将 RVC 作为异步效果接入音频处理链。

## 项目结构

```text
voice_change/
├── main.py                          # 程序入口与顶层资源生命周期
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
├── effects/
│   ├── manager.py                   # 有序效果链
│   ├── ai_voice.py                  # RVC 分块、队列、重叠和拼接适配器
│   ├── gain.py                      # 增益
│   ├── echo.py                      # 回声
│   └── robot.py                     # 机器人音色
├── gui/
│   ├── main_window.py               # 横向双列主窗口与高级参数面板
│   ├── rvc_control_panel.py         # AI 开关、模型导入和实时模式选择
│   ├── self_monitor_panel.py        # 自监听设备与音量控制
│   └── i18n.py                      # English/中文界面文本
├── models/rvc/                      # 项目内模型 profile；二进制不进入 Git
└── tests/                           # 单元、集成、真实模型和 benchmark 测试
```

## 运行逻辑

```text
Microphone
  -> AudioStream
  -> OutputRoutingEffectManager
  -> AIVoiceEffect
  -> GainEffect
  -> EchoEffect
  -> RobotEffect
  -> Main output / VB-CABLE
  -> Optional Self Monitor / Headphones
```

`AudioStream` 的 PortAudio 回调只处理短音频块，不在回调线程内执行耗时的模型推理。`AIVoiceEffect` 将输入累积成 RVC 窗口并提交给 `RVCWorker`；Worker 在独立线程中调用 `RVCEngine`，完成后的结果进入输出缓冲区，再按照音频回调所需长度连续返回。

默认 Balanced 模式使用 500 ms chunk 和 50 ms overlap。相邻推理窗口使用线性 crossfade 拼接，减少 chunk 边界突变。输入和输出队列均有容量上限，推理落后时不会让实时回调无限积压。

启动时 `RVCModelManager` 合并扫描 `models/rvc/` 的内置模型和 `config/user_models.json` 中登记的外部模型。`RVCRuntime` 创建 Engine/Worker、加载所选模型并挂接 `AIVoiceEffect`；模型切换或程序退出时依次停止 Worker、卸载模型并释放 index cache。模型加载失败时，基础效果和非 AI 音频链仍可使用。

界面更新 Pitch、Index Rate、Protect 和 RMS Mix Rate 时，只替换线程安全的推理配置快照，不会重新加载 `.pth`、`.index`、HuBERT、RMVPE 或重建 Pipeline。实时模式切换只安全更新分块缓冲和 Worker 的 chunk shape，不更换已加载模型。

## 使用的算法

### RVC 声音转换

- **HuBERT/ContentVec 内容编码**：提取输入语音的内容特征，降低对原说话人音色的依赖。
- **RMVPE 基频提取**：估计 F0 曲线，为有音高条件的 RVC 模型提供基频信息。profile 也允许选择后端支持的 `pm` 或 `fcpe`。
- **FAISS 特征检索**：从 `.index` 搜索目标声音训练特征，由 Index Rate 控制检索特征与输入特征的混合比例；没有 index 时可使用无 index 模式。
- **RVC Generator/Synthesizer**：组合内容特征、F0、说话人信息和检索特征，合成目标音色波形。
- **RMS envelope mixing**：以 RMS Mix Rate 控制输出响度包络对输入响度的跟随程度。
- **Protect**：保护清辅音、气息等低能量区域，降低过度转换造成的音素损坏。

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

## 怎样启用

### 1. 安装环境

使用 Windows、Python 3.11 和支持 CUDA 12.1 的 NVIDIA 驱动，安装项目依赖：

```powershell
cd D:\Project_all\voice_change
E:\Anaconda\envs\voice_change\python.exe -m pip install -r requirements.txt
```

CPU 可以启动基础功能，但实时 RVC 推荐使用 NVIDIA GPU。实际验证的环境与依赖版本记录在 `requirements.txt`。

### 2. 配置 RVC 后端

在 `config/settings.py` 中设置：

```python
RVC_SOURCE_DIR = r"D:\path\to\rvc_source"
RVC_MODELS_DIR = r"D:\path\to\backend_models"
```

`RVC_SOURCE_DIR` 必须指向兼容的外部 RVC 源码目录；`RVC_MODELS_DIR` 至少应包含：

```text
backend_models/
├── hubert/
│   ├── config.json
│   └── pytorch_model.bin
└── rmvpe/
    └── rmvpe.pt
```

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
cd D:\Project_all\voice_change
E:\Anaconda\envs\voice_change\python.exe main.py
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
