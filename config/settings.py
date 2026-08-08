from __future__ import annotations

import os
import sys
from pathlib import Path

from config.rvc_realtime import (
    RVC_DEFAULT_REALTIME_PRESET,
    get_rvc_realtime_preset,
)

"""全局配置"""

# ---- 音频参数 ----
SAMPLE_RATE: int = 48000          # 采样率 (Hz)，统一麦克风、VB-CABLE 与监听链路
CHANNELS: int = 1                  # 单声道
BLOCKSIZE: int = 256               # 每回调帧数，越小延迟越低 (128/256/512)
DTYPE: str = "float32"             # sounddevice 数据类型
LATENCY: str = "low"               # "low" 让 PortAudio 选设备最低延迟

# ---- 应用 ----
LOG_LEVEL: str = "DEBUG"
APP_NAME: str = "Voice Changer"
APP_VERSION: str = "2.0.0"

# ---- 设备配置 ----
INPUT_DEVICE: int | None = None            # 输入设备索引，None = 系统默认
OUTPUT_DEVICE: str = "VB-CABLE"            # 输出设备关键字，"VB-CABLE" = 自动查找
SHOW_DEVICE_LIST: bool = False             # True = 启动时打印设备列表，False = 跳过
AUTO_SELECT_DEVICES: bool = True           # True = 自动选设备，False = 手动选

# ---- 最终输出增益 ----
GAIN_VALUE: float = 1.0

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

def _application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT: Path = _application_root()
BUNDLE_ROOT: Path = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT)).resolve()
IS_FROZEN: bool = bool(getattr(sys, "frozen", False))
LOCAL_ASSETS_ROOT: Path = PROJECT_ROOT / "local_assets"
_DEFAULT_RVC_MODEL_LIBRARY_DIR = (
    PROJECT_ROOT / "models" / "rvc"
    if IS_FROZEN
    else LOCAL_ASSETS_ROOT / "rvc" / "voice_models"
)
RVC_MODEL_LIBRARY_DIR: str = os.environ.get(
    "VOICE_CHANGE_RVC_VOICE_MODELS_DIR",
    str(_DEFAULT_RVC_MODEL_LIBRARY_DIR),
)
RVC_DEFAULT_MODEL: str = (
    os.environ.get("VOICE_CHANGE_RVC_DEFAULT_MODEL", "modelF").strip() or "modelF"
)
RVC_USER_MODELS_FILE: str = str(PROJECT_ROOT / "config" / "user_models.json")

_DEFAULT_RVC_SOURCE_DIR = (
    BUNDLE_ROOT / "rvc_source"
    if IS_FROZEN
    else PROJECT_ROOT / "rvc_source"
)
_DEFAULT_RVC_MODELS_DIR = (
    PROJECT_ROOT / "rvc_models"
    if IS_FROZEN
    else LOCAL_ASSETS_ROOT / "rvc" / "foundation_models"
)
RVC_SOURCE_DIR: str = os.environ.get(
    "VOICE_CHANGE_RVC_SOURCE_DIR", str(_DEFAULT_RVC_SOURCE_DIR)
)
RVC_MODELS_DIR: str = os.environ.get(
    "VOICE_CHANGE_RVC_MODELS_DIR", str(_DEFAULT_RVC_MODELS_DIR)
)
# RVC voice directory (contains .pth + .index files)
RVC_VOICE_DIR: str = str(Path(RVC_MODEL_LIBRARY_DIR) / RVC_DEFAULT_MODEL)

# ---- Optional external Beatrice v2 backend ----
LOCAL_SETTINGS_FILE: str = str(PROJECT_ROOT / "config" / "local_settings.json")
BEATRICE_MODEL_LIBRARY_DIR: str = str(PROJECT_ROOT / "models" / "beatrice")
BEATRICE_ENV_MODELS_DIR: str | None = (
    os.environ.get("VOICE_CHANGE_BEATRICE_MODELS_DIR", "").strip() or None
)
BEATRICE_DEFAULT_RUNTIME_DIR: str = str(PROJECT_ROOT / "third_party" / "beatrice_runtime")
BEATRICE_RUNTIME_DIR: str | None = (
    os.environ.get("VOICE_CHANGE_BEATRICE_RUNTIME_DIR", "").strip() or None
)
BEATRICE_CALLBACK_SIZE: int = BLOCKSIZE
BEATRICE_INPUT_QUEUE_SIZE: int = 8
BEATRICE_STARTUP_BUFFER_SIZE: int = 512
BEATRICE_WORKER_STOP_TIMEOUT: float = 5.0
