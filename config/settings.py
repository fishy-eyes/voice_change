from __future__ import annotations

from pathlib import Path

from config.rvc_realtime import (
    RVC_DEFAULT_REALTIME_PRESET,
    get_rvc_realtime_preset,
)

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
ENABLE_GAIN: bool = False                   # 启用增益效果
GAIN_VALUE: float = 2.0                    # 增益倍数

ENABLE_ECHO: bool = False                  # 启用回声效果
ECHO_DELAY: float = 0.3                    # 回声延迟 (秒)
ECHO_DECAY: float = 0.4                    # 回声衰减系数

ENABLE_ROBOT: bool = False                 # 启用机器人音效
ROBOT_FREQUENCY: int = 80                  # 机器人音效频率 (Hz)

# ---- AI 变声 ----
ENABLE_AI_VOICE: bool = True
RVC_REALTIME_PRESET: str = RVC_DEFAULT_REALTIME_PRESET
_RVC_REALTIME_DEFAULT = get_rvc_realtime_preset(RVC_REALTIME_PRESET)
RVC_CHUNK_MS: int = _RVC_REALTIME_DEFAULT.chunk_ms
RVC_OVERLAP_MS: int = _RVC_REALTIME_DEFAULT.overlap_ms
RVC_CHUNK_SIZE: int = _RVC_REALTIME_DEFAULT.chunk_samples(SAMPLE_RATE)
RVC_OVERLAP_SIZE: int = _RVC_REALTIME_DEFAULT.overlap_samples(SAMPLE_RATE)
RVC_INPUT_QUEUE_SIZE: int = 2
RVC_WARMUP_ENABLED: bool = True
RVC_WARMUP_TIMEOUT: float = 120.0
RVC_WORKER_STOP_TIMEOUT: float = 5.0
RVC_PITCH_SHIFT: int = 0
RVC_F0_METHOD: str = "rmvpe"         # f0 estimation method
RVC_INDEX_RATE: float = 0.75         # index matching rate (0.0 - 1.0)
RVC_RMS_MIX_RATE: float = 0.25       # RMS envelope mix rate
RVC_PROTECT: float = 0.33            # consonant protection (0.0 - 0.5)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RVC_MODEL_LIBRARY_DIR: str = str(PROJECT_ROOT / "models" / "rvc")
RVC_DEFAULT_MODEL: str = "modelF"

# RVC source code root (independent, managed separately)
RVC_SOURCE_DIR: str = r"D:\Project_all\rvc_core_test\rvc_source"
# RVC models root (hubert, rmvpe, voices)
RVC_MODELS_DIR: str = r"D:\Project_all\rvc_core_test\models"
# RVC voice directory (contains .pth + .index files)
RVC_VOICE_DIR: str = str(Path(RVC_MODEL_LIBRARY_DIR) / RVC_DEFAULT_MODEL)
