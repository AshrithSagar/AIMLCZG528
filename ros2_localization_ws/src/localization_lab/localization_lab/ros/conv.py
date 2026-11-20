"""Covariance packing for geometry_msgs/PoseWithCovariance (6x6 over x, y, z, roll, pitch, yaw)."""

from __future__ import annotations

import numpy as np

from sensor_motion_lab.core.geometry import FloatArray


class CovarianceConv:
    IDX = (0, 1, 5)  # x, y, yaw rows/cols in the 6x6 matrix

    @classmethod
    def to_ros(cls, cov3: FloatArray) -> list[float]:
        full = np.zeros((6, 6))
        for a, i in enumerate(cls.IDX):
            for b, j in enumerate(cls.IDX):
                full[i, j] = cov3[a, b]
        return [float(v) for v in full.ravel()]

    @classmethod
    def from_ros(cls, cov36: list[float] | FloatArray) -> FloatArray:
        full = np.asarray(cov36, dtype=np.float64).reshape(6, 6)
        return full[np.ix_(cls.IDX, cls.IDX)]
