"""Versioned JSON persistence for customization profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from loguru import logger

from customization.schemas import CustomizationProfile, RVCParameterSet


CURRENT_PROFILE_VERSION = 1


@dataclass(frozen=True)
class ProfileLoadResult:
    profile: CustomizationProfile | None
    warnings: tuple[str, ...] = ()
    error: str | None = None


class ProfileStore:
    def save(self, profile: CustomizationProfile, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(profile.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(destination)
        logger.info("配置保存路径: {}", destination)
        return destination

    def load(
        self,
        path: str | Path,
        *,
        expected_model_hash: str | None = None,
        available_input_devices: set[str] | None = None,
    ) -> ProfileLoadResult:
        source = Path(path)
        try:
            with source.open("r", encoding="utf-8") as handle:
                values = json.load(handle)
            if not isinstance(values, dict):
                raise TypeError("profile root must be an object")
            profile = CustomizationProfile.from_dict(values)
        except Exception as exc:
            return ProfileLoadResult(None, error=f"{type(exc).__name__}: {exc}")

        warnings: list[str] = []
        if profile.profile_version > CURRENT_PROFILE_VERSION:
            return ProfileLoadResult(
                None,
                error=(
                    f"unsupported profile version {profile.profile_version}; "
                    f"maximum is {CURRENT_PROFILE_VERSION}"
                ),
            )
        if expected_model_hash and profile.model.model_hash != expected_model_hash:
            warnings.append("模型哈希不匹配，配置可能属于其他模型")

        index_available = bool(
            profile.model.index_path and Path(profile.model.index_path).is_file()
        )
        if profile.model.has_index and not index_available:
            warnings.append("index 文件缺失，已降级为 index_rate=0")
            profile = replace(
                profile,
                parameters=replace(profile.parameters, index_rate=0.0),
                model=replace(
                    profile.model,
                    has_index=False,
                    index_loadable=False,
                ),
            )
        if (
            available_input_devices is not None
            and profile.input_device_name not in available_input_devices
        ):
            warnings.append("保存的输入设备当前不可用")

        if warnings:
            profile = replace(profile, warnings=tuple((*profile.warnings, *warnings)))
        return ProfileLoadResult(profile, tuple(warnings))


def ensure_no_index(parameters: RVCParameterSet, *, has_valid_index: bool) -> RVCParameterSet:
    return parameters if has_valid_index else replace(parameters, index_rate=0.0)
