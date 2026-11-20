"""Pinhole camera model and the RGB-D frame container."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .se3 import FloatArray


@dataclass(frozen=True, slots=True)
class PinholeCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    # ----- presets -----------------------------------------------------------
    @classmethod
    def tum_freiburg1(cls) -> PinholeCamera:
        """TUM RGB-D 'freiburg1' (fr1) colour intrinsics, from the dataset's documentation."""
        return cls(517.3, 516.5, 318.6, 255.3, 640, 480)

    @classmethod
    def tum_default(cls) -> PinholeCamera:
        """The 'ROS default' intrinsics TUM uses for its pre-registered depth images."""
        return cls(525.0, 525.0, 319.5, 239.5, 640, 480)

    @classmethod
    def from_k(cls, k: list[float] | FloatArray, width: int, height: int) -> PinholeCamera:
        """From a row-major 3x3 K matrix (sensor_msgs/CameraInfo.k)."""
        k = np.asarray(k, dtype=np.float64).ravel()
        return cls(float(k[0]), float(k[4]), float(k[2]), float(k[5]), int(width), int(height))

    @classmethod
    def from_fov(cls, hfov_rad: float, width: int, height: int) -> PinholeCamera:
        """Square pixels, principal point at the image centre (Gazebo's camera model)."""
        f = (width / 2.0) / np.tan(hfov_rad / 2.0)
        return cls(f, f, (width - 1) / 2.0, (height - 1) / 2.0, width, height)

    # ----- geometry ----------------------------------------------------------
    @property
    def k(self) -> FloatArray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def scaled(self, factor: float) -> PinholeCamera:
        return PinholeCamera(self.fx * factor, self.fy * factor, self.cx * factor, self.cy * factor,
                             int(round(self.width * factor)), int(round(self.height * factor)))

    def backproject(self, uv: FloatArray, depth: FloatArray) -> FloatArray:
        """(N, 2) pixels + (N,) metric depth -> (N, 3) points in the optical frame."""
        x = (uv[:, 0] - self.cx) / self.fx * depth
        y = (uv[:, 1] - self.cy) / self.fy * depth
        return np.column_stack((x, y, depth))

    def project(self, points: FloatArray) -> FloatArray:
        z = points[:, 2]
        return np.column_stack((self.fx * points[:, 0] / z + self.cx, self.fy * points[:, 1] / z + self.cy))

    def ray_directions(self) -> FloatArray:
        """(H, W, 3) unit-depth ray directions (z = 1) for every pixel."""
        u, v = np.meshgrid(np.arange(self.width, dtype=np.float64), np.arange(self.height, dtype=np.float64))
        return np.dstack(((u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)))


@dataclass(frozen=True, slots=True)
class RgbdFrame:
    """One synchronised colour + depth pair. Depth is float32 metres, 0 / NaN = invalid."""

    stamp: float
    rgb: NDArray[np.uint8]  # (H, W, 3), RGB order
    depth: NDArray[np.float32]  # (H, W)

    @property
    def gray(self) -> NDArray[np.uint8]:
        return (0.299 * self.rgb[..., 0] + 0.587 * self.rgb[..., 1] + 0.114 * self.rgb[..., 2]).astype(np.uint8)

    def depth_at(self, uv: FloatArray) -> FloatArray:
        """Nearest-pixel depth lookup; invalid -> 0."""
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, self.depth.shape[1] - 1)
        v = np.clip(np.round(uv[:, 1]).astype(int), 0, self.depth.shape[0] - 1)
        d = self.depth[v, u].astype(np.float64)
        d[~np.isfinite(d)] = 0.0
        return d
