## Voice Changer v1.0.0

首个 Windows 实时 AI 变声器稳定版本。

### 主要功能

- 48 kHz 单声道实时音频链、VB-CABLE 路由和独立耳机自监听。
- 异步 RVC Worker、固定窗口、overlap/crossfade、有限队列和 FAISS index 缓存。
- RVC 模型发现、外部模型导入、安全切换以及 Pitch、Index Rate、Protect、RMS Mix Rate 实时调节。
- 智能音色适配：Pitch 粗调/细调、Index Rate、Protect、RMS Mix Rate 严格逐项选择。
- 完整录音候选试听、整体响度匹配和技术损坏筛选。
- 按模型文件夹命名的 JSON 定制配置及模型/index 兼容性检查。
- Gain、Echo、Robot 基础效果和中英文 GUI。

### Windows 包说明

- 下载 `voice_change-v1.0.0-windows-x64.tar.xz`，使用 Windows 自带的 `tar -xJf` 解压后运行 `VoiceChanger.exe`。
- 包含应用、Python 运行时、PyTorch/CUDA 运行库以及 MIT 许可的 RVC 推理源码。
- 不包含 HuBERT、RMVPE、音色 `.pth` 或 `.index` 权重；请按压缩包内 `README.txt` 放置或导入。
- 可执行文件未进行代码签名，Windows SmartScreen 可能显示提示。
- SHA-256 校验值见同名 `.sha256` 文件。
