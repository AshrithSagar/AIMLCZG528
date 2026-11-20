"""
Shared pieces for every localizer.

W2 is an *overlay* on the W1 workspace: the map, the motion model and the
sensor models are imported from `sensor_motion_lab.core`, not copied.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sensor_motion_lab.core import Angle, GridMap, Pose2D, Scan
from sensor_motion_lab.core.geometry import FloatArray


@dataclass(frozen=True, slots=True)
class PoseEstimate:
    pose: Pose2D
    covariance: FloatArray  # (3, 3) over x, y, theta

    @property
    def position_std(self) -> float:
        return float(math.sqrt(max(np.trace(self.covariance[:2, :2]), 0.0)))

    @property
    def heading_std_deg(self) -> float:
        return float(math.degrees(math.sqrt(max(self.covariance[2, 2], 0.0))))


@dataclass(slots=True)
class UpdateStats:
    predict_ms: float = 0.0
    correct_ms: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return self.predict_ms + self.correct_ms


class Localizer(ABC):
    """Bayes-filter interface shared by grid localization, MCL and the ML variants."""

    name: str = "localizer"

    def __init__(self) -> None:
        self.stats = UpdateStats()

    @abstractmethod
    def reset_global(self) -> None:
        """Uniform belief over the free space (global localization / 'I am lost')."""

    @abstractmethod
    def reset_at(self, pose: Pose2D, std_xy: float = 0.25, std_theta: float = 0.2) -> None:
        """Gaussian belief around a known pose (RViz '2D Pose Estimate')."""

    @abstractmethod
    def predict(self, odom_prev: Pose2D, odom_curr: Pose2D) -> None: ...

    @abstractmethod
    def correct(self, scan: Scan) -> None: ...

    @abstractmethod
    def estimate(self) -> PoseEstimate: ...

    def update(self, odom_prev: Pose2D, odom_curr: Pose2D, scan: Scan) -> PoseEstimate:
        t0 = time.perf_counter()
        self.predict(odom_prev, odom_curr)
        t1 = time.perf_counter()
        self.correct(scan)
        t2 = time.perf_counter()
        self.stats.predict_ms, self.stats.correct_ms = 1000 * (t1 - t0), 1000 * (t2 - t1)
        return self.estimate()


class ArenaMask:
    """
    The enclosed region the robot can be in: the largest connected non-obstacle region that
    does not touch the map border (so the margin *outside* the walls is excluded).
    """

    @staticmethod
    def compute(grid: GridMap) -> NDArray[np.bool_]:
        from scipy.ndimage import label  # noqa: PLC0415

        free = ~grid.occupied
        labels, n = label(free)
        border = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))) - {0}
        sizes = [(int((labels == k).sum()), k) for k in range(1, n + 1) if k not in border]
        return labels == max(sizes)[1] if sizes else free  # open map: every free cell counts


class FreeSpace:
    """Poses the robot can occupy: inside the arena and at least `clearance` from obstacles."""

    def __init__(self, grid: GridMap, clearance: float = 0.2) -> None:
        self.grid = grid
        self.arena = ArenaMask.compute(grid)
        gy, gx = np.nonzero(self.arena & (grid.distance_field >= clearance))
        self.cells_x = grid.origin_x + (gx + 0.5) * grid.resolution
        self.cells_y = grid.origin_y + (gy + 0.5) * grid.resolution
        if self.cells_x.size == 0:
            raise ValueError("map has no free space")

    def sample(self, n: int, rng: np.random.Generator) -> FloatArray:
        idx = rng.integers(0, self.cells_x.size, n)
        jitter = (rng.random((n, 2)) - 0.5) * self.grid.resolution
        return np.column_stack((self.cells_x[idx] + jitter[:, 0], self.cells_y[idx] + jitter[:, 1],
                                rng.uniform(-math.pi, math.pi, n)))

    def contains(self, xy: FloatArray, clearance: float = 0.0) -> NDArray[np.bool_]:
        gx, gy = self.grid.world_to_grid(xy[:, 0], xy[:, 1])
        inside = self.grid.in_bounds(gx, gy)
        out = np.zeros(len(xy), dtype=bool)
        out[inside] = self.arena[gy[inside], gx[inside]]
        return out & (self.grid.distance_to_nearest(xy[:, 0], xy[:, 1], cap=10.0) > clearance)


class WeightedPoses:
    """Mean / covariance of a weighted pose set (circular mean for theta)."""

    @staticmethod
    def estimate(poses: FloatArray, weights: FloatArray) -> PoseEstimate:
        w = weights / weights.sum()
        mx, my = float(w @ poses[:, 0]), float(w @ poses[:, 1])
        th = Angle.circular_mean(poses[:, 2], w)
        d = poses - np.array([mx, my, th])
        d[:, 2] = Angle.wrap_array(d[:, 2])
        cov = (d * w[:, None]).T @ d
        return PoseEstimate(Pose2D(mx, my, th), cov)


class W1Assets:
    """Locates W1's world.yaml / robot.yaml (installed share/ first, then the source tree)."""

    @staticmethod
    def config_dir() -> Path:
        try:
            from ament_index_python.packages import get_package_share_directory  # noqa: PLC0415

            return Path(get_package_share_directory("mobile_bot_description")) / "config"
        except Exception:  # noqa: BLE001 - not sourced: fall back to the source tree next to sensor_motion_lab
            import sensor_motion_lab  # noqa: PLC0415

            cand = Path(sensor_motion_lab.__file__).resolve().parents[2] / "mobile_bot_description" / "config"
            if cand.is_dir():
                return cand
            raise FileNotFoundError("cannot find mobile_bot_description/config (source the W1 workspace)")
