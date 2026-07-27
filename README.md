# Voice Changer

Windows 平台实时变声器。

## 功能规划

- 麦克风实时采集与播放
- DSP 音效：变调、回声、混响、机器人音
- PySide6 图形界面
- 预留 AI 变声模型接口

## 项目结构

```
voice_change/
├── main.py            # 程序入口
├── config/            # 全局配置
├── audio/             # 音频采集、播放、流管理
├── effects/           # DSP 音效处理
├── gui/               # 图形界面
├── utils/             # 工具函数
├── models/            # AI 模型（预留）
├── tests/             # 测试
└── assets/            # 图标、音效素材
```

## 运行

```bash
pip install -r requirements.txt
python main.py
```
