"""
Line features for the feature-based measurement model (Probabilistic Robotics §6.6).

  LineFeature     a line in normal form (rho, alpha): the set of points with
                  x cos(alpha) + y sin(alpha) = rho, rho >= 0, plus its support
  LineExtractor   scan endpoints -> line features (sequential RANSAC + total least squares)
  LineMap         the feature map: every face of every box in world.yaml as a line segment

Raw returns have no identity; a *feature* does. Extraction turns 360 ranges into
a handful of walls, which is what makes data association (and a compact map) possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import FloatArray
from .world import WorldSpec


@dataclass(frozen=True, slots=True)
class LineFeature:
    rho: float  # perpendicular distance from the origin (m), >= 0
    alpha: float  # direction of the normal (rad)
    support: int  # number of points that produced it
    p0: tuple[float, float]  # segment end points (extent of the supporting points)
    p1: tuple[float, float]

    @classmethod
    def fit(cls, pts: FloatArray) -> LineFeature:
        """Total least squares (orthogonal regression) through the points."""
        c = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - c)
        normal = vt[-1]
        rho = float(normal @ c)
        if rho < 0:
            normal, rho = -normal, -rho
        alpha = math.atan2(normal[1], normal[0])
        direction = np.array([-normal[1], normal[0]])
        t = (pts - c) @ direction
        p0, p1 = c + direction * t.min(), c + direction * t.max()
        return cls(rho, alpha, len(pts), (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])))

    @classmethod
    def through(cls, a: FloatArray, b: FloatArray) -> LineFeature:
        return cls.fit(np.vstack((a, b)))

    def distances(self, pts: FloatArray) -> FloatArray:
        n = np.array([math.cos(self.alpha), math.sin(self.alpha)])
        return np.abs(pts @ n - self.rho)


@dataclass(frozen=True, slots=True)
class LineExtractorParams:
    dist_tol: float = 0.03  # a point belongs to a line if it is closer than this (m)
    min_points: int = 8  # smaller groups are not trusted as features
    max_lines: int = 8
    candidates: int = 300  # RANSAC hypotheses per line (all pairs if fewer)
    seed: int = 0


class LineExtractor:
    """
    Sequential RANSAC: find the line with most inliers, refit it by total least squares,
    remove its points, repeat. A point that fits no line (e.g. a person in front of a wall)
    is left over as an outlier instead of dragging a least-squares fit away.
    """

    def __init__(self, params: LineExtractorParams = LineExtractorParams()) -> None:
        self.params = params

    def extract(self, pts: FloatArray) -> list[LineFeature]:
        p = self.params
        rng = np.random.default_rng(p.seed)
        remaining = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        lines: list[LineFeature] = []
        while len(remaining) >= max(p.min_points, 2) and len(lines) < p.max_lines:
            best_inliers = self._best_inliers(remaining, rng)
            if best_inliers.sum() < p.min_points:
                break
            line = LineFeature.fit(remaining[best_inliers])
            lines.append(line)
            remaining = remaining[~best_inliers]
        return lines

    def hypotheses(self, pts: FloatArray, rng: np.random.Generator | None = None) -> list[np.ndarray]:
        """Inlier masks of every two-point line hypothesis (a random subset if there are many)."""
        n = len(pts)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if len(pairs) > self.params.candidates:
            rng = rng or np.random.default_rng(self.params.seed)
            pairs = [tuple(rng.choice(n, 2, replace=False)) for _ in range(self.params.candidates)]
        return [LineFeature.through(pts[i], pts[j]).distances(pts) < self.params.dist_tol
                for i, j in pairs if not np.allclose(pts[i], pts[j])]

    def _best_inliers(self, pts: FloatArray, rng: np.random.Generator) -> np.ndarray:
        best, best_key = np.zeros(len(pts), dtype=bool), (0, 0.0)
        for inl in self.hypotheses(pts, rng):
            key = (int(inl.sum()), self._extent(pts[inl]))  # ties -> the longer segment
            if key > best_key:
                best, best_key = inl, key
        return best

    @staticmethod
    def _extent(pts: FloatArray) -> float:
        return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) if len(pts) else 0.0


class LineMap:
    """The feature map M: every face of every box (walls and obstacles) as a segment in the map frame."""

    def __init__(self, segments: FloatArray) -> None:
        self.segments = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 2)  # (M, 2 ends, xy)

    @classmethod
    def from_world(cls, world: WorldSpec) -> LineMap:
        segs = []
        for box in world.all_boxes():
            x0, y0, x1, y1 = box.bounds
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            segs += [(corners[k], corners[(k + 1) % 4]) for k in range(4)]
        return cls(np.array(segs))

    def in_sensor_frame(self, poses: FloatArray) -> tuple[FloatArray, FloatArray]:
        """(rho, alpha) of every map line as seen from each sensor pose (x, y, theta): the predicted features.

        poses (N, 3) -> rho, alpha of shape (N, M)."""
        poses = np.atleast_2d(poses)
        c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
        rel = self.segments[None] - poses[:, None, None, :2]  # (N, M, 2, 2)
        lx = c[:, None, None] * rel[..., 0] + s[:, None, None] * rel[..., 1]  # world -> sensor
        ly = -s[:, None, None] * rel[..., 0] + c[:, None, None] * rel[..., 1]
        dx, dy = lx[..., 1] - lx[..., 0], ly[..., 1] - ly[..., 0]
        length = np.hypot(dx, dy)
        nx, ny = -dy / length, dx / length
        rho = nx * lx[..., 0] + ny * ly[..., 0]
        sign = np.where(rho < 0, -1.0, 1.0)
        return np.abs(rho), np.arctan2(ny * sign, nx * sign)
