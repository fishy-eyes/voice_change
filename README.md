# Voice Changer

Windows 实时 AI 变声器。

## 项目介绍

基于 RVC（Retrieval-based Voice Conversion）的 Windows 实时 AI 变声器，用于个人娱乐和分享。

技术路线：

```
麦克风输入 → 音频采集 → 音效处理 → RVC AI变声 → 虚拟麦克风(VB-CABLE)输出
```

## 当前完成状态

### 已完成

- 音频采集框架（sounddevice duplex stream）
- 音效系统（增益、回声、机器人音效 + 效果链管理）
- AI 变声适配层（AIVoiceEffect 封装 RVCEngine）
- RVC 离线推理集成（模型加载 / HuBERT / RMVPE / Pipeline / 推理）
- PySide6 图形界面基础框架
- 离线推理冒烟测试通过

### 未完成

- 实时 RVC 推理（缓冲区管理、低延迟优化）
- 模型管理 GUI（切换音色、参数调节）
- 延迟优化（GPU pipeline 优化、流式推理）

## 项目结构

```
voice_change/
├── main.py                 # 程序入口
├── config/                 # 全局配置（settings.py）
├── audio/                  # 音频采集、播放、流管理
│   ├── stream.py           # duplex 音频流
│   ├── recorder.py         # 麦克风采集
│   ├── player.py           # 音频播放
│   └── device_manager.py   # 设备枚举
├── effects/                # 音效处理
│   ├── base.py             # BaseEffect 抽象接口
│   ├── manager.py          # EffectManager 效果链
│   ├── ai_voice.py         # AIVoiceEffect（RVCEngine 适配器）
│   ├── gain.py / echo.py / robot.py
│   └── ...
├── ai/                     # AI 推理引擎
│   └── rvc_engine.py       # RVCEngine（真实 RVC 推理封装）
├── gui/                    # PySide6 图形界面
├── core/                   # 运行时上下文
├── utils/                  # 工具函数
├── tests/                  # 测试
│   ├── test_rvc_integration.py  # RVC 集成冒烟测试
│   └── assets/             # 测试音频
└── assets/                 # 图标、音效素材
```

## RVC 环境说明

RVC 源码和模型独立管理，不在本项目目录内。

### RVC 源码

```
D:\Project_all\rvc_core_test\rvc_source
```

### 模型目录

```
D:\Project_all\rvc_core_test\models
├── hubert/          # HuBERT 特征提取模型（Transformers 格式）
├── rmvpe/           # RMVPE 基频估计模型
│   └── rmvpe.pt
└── voices/          # 音色模型
    ├── *.pth        # 音色权重（SynthesizerTrn checkpoint）
    └── *.index      # FAISS 检索索引（可选，提升音色相似度）
```

### 当前测试模型

- 音色：`VT-TTS_Hikari.pth`
- 索引：`added_IVF1344_Flat_nprobe_1_VT-TTS_Hikari_v2.index`
- 版本：RVC v2 / f0 / 48kHz

## 环境依赖

```bash
conda activate voice_change

# 核心依赖
torch==2.1.2+cu121
torchaudio==2.1.2
numpy==1.26.4
librosa==0.11.0
faiss-cpu==1.14.3
soundfile==0.14.0
praat-parselmouth==0.4.7
transformers==4.49.0
loguru
scipy
PySide6
sounddevice
```

## 测试方法

### RVC 集成冒烟测试

```bash
conda activate voice_change
python -u tests\test_rvc_integration.py
```

测试内容：模型加载 → HuBERT → Pipeline → 推理 → 输出 wav

预计耗时：40-60 秒

输出文件：`tests\assets\rvc_output.wav`

### 运行主程序

```bash
conda activate voice_change
python main.py
```

## 效果链

当前效果链顺序：

```
Mic → AIVoiceEffect → GainEffect → EchoEffect → RobotEffect → VB-CABLE
```

`AIVoiceEffect` 位于链首，后续 DSP 效果作用于 AI 变声后的音频。
