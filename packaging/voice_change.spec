# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


project_root = Path(SPECPATH).parent
rvc_source = Path(os.environ["VOICE_CHANGE_BUILD_RVC_SOURCE_DIR"]).resolve()
if not (rvc_source / "LICENSE").is_file():
    raise SystemExit(f"RVC source LICENSE not found: {rvc_source}")

sys.path.insert(0, str(rvc_source))
console_build = os.environ.get("VOICE_CHANGE_BUILD_CONSOLE") == "1"

datas = [
    (str(rvc_source / "infer"), "rvc_source/infer"),
    (str(rvc_source / "tools"), "rvc_source/tools"),
    (str(rvc_source / "configs"), "rvc_source/configs"),
    (str(rvc_source / "i18n"), "rvc_source/i18n"),
    (str(rvc_source / "LICENSE"), "licenses"),
]
binaries = []
hiddenimports = [
    "torch",
    "torchaudio",
    "transformers",
    "transformers.models.hubert",
    "transformers.models.hubert.configuration_hubert",
    "transformers.models.hubert.modeling_hubert",
    "faiss",
    "librosa",
    "parselmouth",
    "infer.hubert",
    "infer.module.models",
    "infer.rmvpe",
    "infer.vc.pipeline",
    "tools.cuda_graph",
    "configs.config",
]
hiddenimports += collect_submodules("transformers.models.hubert")

sounddevice_datas, sounddevice_binaries, sounddevice_hidden = collect_all(
    "_sounddevice_data"
)
datas += sounddevice_datas
binaries += sounddevice_binaries
hiddenimports += sounddevice_hidden

conda_bz2 = Path(sys.prefix) / "Library" / "bin" / "libbz2.dll"
if conda_bz2.is_file():
    binaries.append((str(conda_bz2), "."))

shiboken_dll = Path(sys.prefix) / "Lib" / "site-packages" / "shiboken6" / "shiboken6.abi3.dll"
if shiboken_dll.is_file():
    binaries.append((str(shiboken_dll), "PySide6"))

python_abi_dll = Path(sys.prefix) / "python3.dll"
if python_abi_dll.is_file():
    binaries.append((str(python_abi_dll), "PySide6"))

a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root), str(rvc_source)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(project_root / "packaging" / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest"],
    noarchive=False,
    optimize=0,
)

# PyInstaller may find an incompatible ICU from the base Conda installation.
# Qt 6.11 on current Windows uses the operating-system ICU instead.
base_conda_icu = {"icuuc.dll", "icudt73.dll"}
a.binaries = [entry for entry in a.binaries if Path(entry[0]).name.lower() not in base_conda_icu]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VoiceChanger",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=console_build,
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=str(project_root / "packaging" / "version_info.txt"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="VoiceChanger-v1.0.0",
)
