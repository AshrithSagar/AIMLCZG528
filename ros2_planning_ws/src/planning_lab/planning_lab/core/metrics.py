"""Path quality metrics (shared by the benchmark and the ROS planner node)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sensor_motion_lab.core.geometry import FloatArray

from .grid import PlanningGrid


@dataclass(frozen=True, slots=True)
class PathQuality:
    length: float
    min_clearance: float  # m from the path to the nearest obstacle surface (static + moving)
    turning_deg: float  # total absolute heading change along the path (smoothness)

    @classmethod
    def of(cls, path: FloatArray, grid: PlanningGrid) -> PathQuality:
        path = np.asarray(path, dtype=np.float64)
        if len(path) < 2:
            return cls(0.0, math.nan, 0.0)
        seg = np.diff(path, axis=0)
        lengths = np.linalg.norm(seg, axis=1)
        dense = np.vstack([path[i] + np.linspace(0, 1, max(2, int(L / 0.02)))[:, None] * seg[i]
                           for i, L in enumerate(lengths) if L > 0] or [path])
        heading = np.arctan2(seg[:, 1], seg[:, 0])[lengths > 1e-9]
        turns = np.abs(np.arctan2(np.sin(np.diff(heading)), np.cos(np.diff(heading)))).sum() if len(heading) > 1 else 0.0
        return cls(float(lengths.sum()), float(grid.clearance(dense).min()), float(math.degrees(turns)))
