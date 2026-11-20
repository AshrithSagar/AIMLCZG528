"""Access to the W1 underlay's configuration (world.yaml / robot.yaml)."""

from __future__ import annotations

from pathlib import Path

from sensor_motion_lab.core.config import DescriptionConfig


class W1Assets:
    @staticmethod
    def config_dir() -> Path:
        return DescriptionConfig.locate()
