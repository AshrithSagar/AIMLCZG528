"""
Rigid-body transforms in 3-D (SE(3)) and rotation helpers.

Convention: `T_a_b` maps points expressed in frame b into frame a,
    p_a = T_a_b * p_b.
Camera frames follow REP-103 *optical* convention (z forward, x right, y down).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class SO3:
    """Rotation-matrix helpers (exp/log maps, quaternions). Namespaced, no free functions."""

    @staticmethod
    def hat(w: FloatArray) -> FloatArray:
        return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])

    @classmethod
    def exp(cls, w: FloatArray) -> FloatArray:
        theta = float(np.linalg.norm(w))
        if theta < 1e-10:
            return np.eye(3) + cls.hat(w)
        k = cls.hat(w / theta)
        return np.eye(3) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)

    @staticmethod
    def log(r: FloatArray) -> FloatArray:
        cos = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
        theta = math.acos(cos)
        if theta < 1e-10:
            return np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2.0
        if math.pi - theta < 1e-6:  # near pi: use the symmetric part
            m = (r + np.eye(3)) / 2.0
            axis = np.sqrt(np.maximum(np.diag(m), 0.0))
            i = int(np.argmax(axis))
            axis = m[:, i] / max(axis[i], 1e-12)
            return theta * axis / np.linalg.norm(axis)
        w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
        return theta * w / (2.0 * math.sin(theta))

    @staticmethod
    def to_quaternion(r: FloatArray) -> FloatArray:
        """(x, y, z, w), w >= 0."""
        tr = np.trace(r)
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
        else:
            i = int(np.argmax(np.diag(r)))
            j, k = (i + 1) % 3, (i + 2) % 3
            s = math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
            q = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / s]
            q[i] = 0.25 * s
            q[j] = (r[j, i] + r[i, j]) / s
            q[k] = (r[k, i] + r[i, k]) / s
        qa = np.asarray(q)
        qa /= np.linalg.norm(qa)
        return qa if qa[3] >= 0 else -qa

    @staticmethod
    def from_quaternion(q: FloatArray) -> FloatArray:
        x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    @staticmethod
    def angle(r: FloatArray) -> float:
        return math.acos(float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))


@dataclass(frozen=True, slots=True)
class SE3:
    rotation: FloatArray  # (3, 3)
    translation: FloatArray  # (3,)

    # ----- construction ----------------------------------------------------
    @classmethod
    def identity(cls) -> SE3:
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_matrix(cls, m: FloatArray) -> SE3:
        return cls(np.array(m[:3, :3], dtype=np.float64), np.array(m[:3, 3], dtype=np.float64))

    @classmethod
    def from_rvec_tvec(cls, rvec: FloatArray, tvec: FloatArray) -> SE3:
        return cls(SO3.exp(np.ravel(rvec)), np.ravel(tvec).astype(np.float64))

    @classmethod
    def from_quaternion(cls, q_xyzw: FloatArray, t: FloatArray) -> SE3:
        return cls(SO3.from_quaternion(q_xyzw), np.asarray(t, dtype=np.float64))

    @classmethod
    def exp(cls, xi: FloatArray) -> SE3:
        """Small-perturbation parameterisation used by the optimiser: (rot vec, translation)."""
        return cls(SO3.exp(xi[:3]), np.asarray(xi[3:], dtype=np.float64))

    # ----- algebra ---------------------------------------------------------
    def __matmul__(self, other: SE3) -> SE3:
        return SE3(self.rotation @ other.rotation, self.rotation @ other.translation + self.translation)

    def inverse(self) -> SE3:
        rt = self.rotation.T
        return SE3(rt, -rt @ self.translation)

    def apply(self, points: FloatArray) -> FloatArray:
        """Transform (N, 3) points."""
        return points @ self.rotation.T + self.translation

    def log(self) -> FloatArray:
        return np.concatenate((SO3.log(self.rotation), self.translation))

    # ----- conversion ------------------------------------------------------
    def matrix(self) -> FloatArray:
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = self.rotation, self.translation
        return m

    def quaternion(self) -> FloatArray:
        return SO3.to_quaternion(self.rotation)

    def rvec(self) -> FloatArray:
        return SO3.log(self.rotation)

    @property
    def rotation_angle(self) -> float:
        return SO3.angle(self.rotation)

    @property
    def translation_norm(self) -> float:
        return float(np.linalg.norm(self.translation))
