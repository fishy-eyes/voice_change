"""全局配置"""

# ---- 音频参数 ----
SAMPLE_RATE: int = 44100          # 采样率 (Hz)，匹配大多数 Windows 设备默认值
CHANNELS: int = 1                  # 单声道
BLOCKSIZE: int = 256               # 每回调帧数，越小延迟越低 (128/256/512)
DTYPE: str = "float32"             # sounddevice 数据类型
LATENCY: str = "low"               # "low" 让 PortAudio 选设备最低延迟

# ---- 应用 ----
LOG_LEVEL: str = "DEBUG"
APP_NAME: str = "Voice Changer"
