"""
SE(2) geometry primitives.

Pure NumPy; no ROS imports, so everything here runs (and is unit-tested)
without a ROS installation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class Angle:
    """Angle helpers (namespaced, no free functions)."""

    TWO_PI: Final[float] = 2.0 * math.pi

    @staticmethod
    def wrap(theta: float) -> float:
        """Wrap a scalar angle to (-pi, pi]."""
        return math.atan2(math.sin(theta), math.cos(theta))

    @staticmethod
    def wrap_array(theta: FloatArray) -> FloatArray:
        """Wrap an array of angles to (-pi, pi]."""
        return np.arctan2(np.sin(theta), np.cos(theta))

    @staticmethod
    def circular_mean(theta: FloatArray, weights: FloatArray | None = None) -> float:
        """Mean of angles (does not break at the +-pi seam)."""
        w = np.ones_like(theta) if weights is None else weights
        return math.atan2(float(np.sum(w * np.sin(theta))), float(np.sum(w * np.cos(theta))))


@dataclass(frozen=True, slots=True)
class Pose2D:
    """A planar pose (x, y, theta) — an element of SE(2)."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    # ----- construction / conversion -------------------------------------
    @classmethod
    def from_array(cls, arr: FloatArray) -> Pose2D:
        return cls(float(arr[0]), float(arr[1]), Angle.wrap(float(arr[2])))

    def as_array(self) -> FloatArray:
        return np.array([self.x, self.y, self.theta], dtype=np.float64)

    # ----- group operations ----------------------------------------------
    def compose(self, other: Pose2D) -> Pose2D:
        """self ⊕ other : express `other` (given in self's frame) in the parent frame."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        return Pose2D(
            self.x + c * other.x - s * other.y,
            self.y + s * other.x + c * other.y,
            Angle.wrap(self.theta + other.theta),
        )

    def inverse(self) -> Pose2D:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return Pose2D(
            -c * self.x - s * self.y,
            s * self.x - c * self.y,
            Angle.wrap(-self.theta),
        )

    def between(self, other: Pose2D) -> Pose2D:
        """Relative motion from self to other, expressed in self's frame: self⁻¹ ⊕ other."""
        return self.inverse().compose(other)

    def distance_to(self, other: Pose2D) -> float:
        return math.hypot(other.x - self.x, other.y - self.y)


class PoseArrayOps:
    """Vectorised helpers for an (N, 3) array of poses [x, y, theta]."""

    @staticmethod
    def mean(poses: FloatArray) -> Pose2D:
        return Pose2D(
            float(np.mean(poses[:, 0])),
            float(np.mean(poses[:, 1])),
            Angle.circular_mean(poses[:, 2]),
        )

    @staticmethod
    def position_covariance(poses: FloatArray) -> FloatArray:
        """2x2 sample covariance of the (x, y) components."""
        return np.cov(poses[:, :2].T)
