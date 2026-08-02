"""Collect librosa source files so Numba cache locators work when frozen."""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


datas = collect_data_files("librosa", excludes=["**/__pycache__"])
hiddenimports = collect_submodules("librosa")
module_collection_mode = "py"
