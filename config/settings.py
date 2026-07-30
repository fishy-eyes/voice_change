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

# ---- 设备配置 ----
INPUT_DEVICE: int | None = None            # 输入设备索引，None = 系统默认
OUTPUT_DEVICE: str = "VB-CABLE"            # 输出设备关键字，"VB-CABLE" = 自动查找
SHOW_DEVICE_LIST: bool = False             # True = 启动时打印设备列表，False = 跳过
AUTO_SELECT_DEVICES: bool = True           # True = 自动选设备，False = 手动选

# ---- 效果器配置 ----
ENABLE_GAIN: bool = True                   # 启用增益效果
GAIN_VALUE: float = 2.0                    # 增益倍数

ENABLE_ECHO: bool = True                  # 启用回声效果
ECHO_DELAY: float = 0.3                    # 回声延迟 (秒)
ECHO_DECAY: float = 0.4                    # 回声衰减系数

ENABLE_ROBOT: bool = True                  # 启用机器人音效
ROBOT_FREQUENCY: int = 80                  # 机器人音效频率 (Hz)
