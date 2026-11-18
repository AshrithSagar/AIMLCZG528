"""Typed view over `mobile_bot_description/config/robot.yaml`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .motion import DriveCalibrationError, DriveGeometry
from .simulation import LidarConfig


@dataclass(frozen=True, slots=True)
class RobotConfig:
    drive: DriveGeometry
    calibration: DriveCalibrationError
    lidar: LidarConfig
    lidar_z: float
    odom_rate_hz: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> RobotConfig:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RobotConfig:
        w, li, cal = raw["wheel"], raw["lidar"], raw.get("calibration", {})
        return cls(
            drive=DriveGeometry(float(w["radius"]), float(w["separation"])),
            calibration=DriveCalibrationError(
                float(cal.get("diameter_ratio", 1.0)), float(cal.get("baseline_ratio", 1.0))
            ),
            lidar=LidarConfig(
                samples=int(li["samples"]),
                range_min=float(li["range_min"]),
                range_max=float(li["range_max"]),
                noise_std=float(li["noise_stddev"]),
                rate_hz=float(li["rate_hz"]),
            ),
            lidar_z=float(li["z_offset"]),
            odom_rate_hz=float(raw.get("odometry", {}).get("rate_hz", 50.0)),
        )
