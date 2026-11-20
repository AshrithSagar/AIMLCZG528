"""
Planning grid built from W1's map (the same `world.yaml`), with:

  * inflation by the robot radius (lethal), like Nav2's inflation layer,
  * a soft cost band near obstacles (paths prefer clearance),
  * moving round obstacles stamped on top (re-stamped as they move -> D* Lite edge changes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sensor_motion_lab.core import GridMap
from sensor_motion_lab.core.geometry import FloatArray

Cell = tuple[int, int]  # (row, col)


@dataclass(frozen=True, slots=True)
class CostConfig:
    resolution: float = 0.1
    robot_radius: float = 0.25  # lethal inflation
    soft_band: float = 0.4  # metres beyond the lethal zone with extra cost
    soft_weight: float = 2.0  # cost multiplier at the edge of the lethal zone (decays to 1)
    obstacle_radius: float = 0.25  # moving obstacles (people)


class PlanningGrid:
    """8-connected cost grid. `lethal[r, c]` = impassable, `cost[r, c]` >= 1 = traversal multiplier."""

    NEIGHBOURS: tuple[tuple[int, int, float], ...] = tuple(
        (dr, dc, math.hypot(dr, dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
    )

    def __init__(self, world: GridMap, config: CostConfig = CostConfig()) -> None:
        self.cfg = config
        self.res = config.resolution
        self.x0, self.y0 = world.origin_x, world.origin_y
        self.cols = int(round(world.width * world.resolution / self.res))
        self.rows = int(round(world.height * world.resolution / self.res))
        xs = self.x0 + (np.arange(self.cols) + 0.5) * self.res
        ys = self.y0 + (np.arange(self.rows) + 0.5) * self.res
        cx, cy = np.meshgrid(xs, ys)
        self.centres = np.stack((cx, cy), axis=-1)  # (R, C, 2)
        self.static_dist = world.distance_to_nearest(cx.ravel(), cy.ravel(), cap=10.0).reshape(self.rows, self.cols)
        self.static_lethal = self.static_dist < config.robot_radius
        self.static_cost = self._soft_cost(self.static_dist)
        self.obstacles = np.zeros((0, 2))
        self.lethal = self.static_lethal.copy()
        self.cost = self.static_cost.copy()

    # ----- coordinates ------------------------------------------------------- #
    def to_cell(self, x: float, y: float) -> Cell:
        return int((y - self.y0) // self.res), int((x - self.x0) // self.res)

    def to_world(self, cell: Cell) -> tuple[float, float]:
        r, c = cell
        return self.x0 + (c + 0.5) * self.res, self.y0 + (r + 0.5) * self.res

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and not self.lethal[cell]

    def is_free_xy(self, xy: FloatArray) -> NDArray[np.bool_]:
        r = ((xy[:, 1] - self.y0) // self.res).astype(int)
        c = ((xy[:, 0] - self.x0) // self.res).astype(int)
        ok = (r >= 0) & (r < self.rows) & (c >= 0) & (c < self.cols)
        out = np.zeros(len(xy), dtype=bool)
        out[ok] = ~self.lethal[r[ok], c[ok]]
        return out

    # ----- moving obstacles ----------------------------------------------------- #
    def set_obstacles(self, centres: FloatArray) -> set[Cell]:
        """Re-stamp moving obstacles; returns the cells whose lethal state changed (for D* Lite)."""
        centres = np.asarray(centres, dtype=np.float64).reshape(-1, 2)
        lethal, cost = self.static_lethal.copy(), self.static_cost.copy()
        if len(centres):
            d = np.min(np.linalg.norm(self.centres[:, :, None, :] - centres[None, None], axis=-1), axis=-1)
            d = d - self.cfg.obstacle_radius
            lethal |= d < self.cfg.robot_radius
            cost = np.maximum(cost, self._soft_cost(d))
        diff = (lethal != self.lethal) | (np.abs(cost - self.cost) > 1e-9)  # D* Lite needs every edge-cost change
        changed = {(int(r), int(c)) for r, c in zip(*np.nonzero(diff))}
        self.obstacles, self.lethal, self.cost = centres, lethal, cost
        return changed

    # ----- collision checking ----------------------------------------------------- #
    def segment_free(self, a: FloatArray, b: FloatArray) -> bool:
        n = max(2, int(math.ceil(np.linalg.norm(np.subtract(b, a)) / (self.res / 2))))
        t = np.linspace(0.0, 1.0, n)[:, None]
        return bool(self.is_free_xy(np.asarray(a) + t * (np.asarray(b) - np.asarray(a))).all())

    def path_free(self, path: FloatArray, from_index: int = 0, horizon: float = math.inf) -> bool:
        travelled = 0.0
        for i in range(from_index, len(path) - 1):
            if not self.segment_free(path[i], path[i + 1]):
                return False
            travelled += float(np.linalg.norm(path[i + 1] - path[i]))
            if travelled > horizon:
                break
        return True

    def cost_xy(self, xy: FloatArray) -> FloatArray:
        """Traversal multiplier at each point (lethal or outside -> large)."""
        r = ((xy[:, 1] - self.y0) // self.res).astype(int)
        c = ((xy[:, 0] - self.x0) // self.res).astype(int)
        ok = (r >= 0) & (r < self.rows) & (c >= 0) & (c < self.cols)
        out = np.full(len(xy), 1e3)
        out[ok] = np.where(self.lethal[r[ok], c[ok]], 1e3, self.cost[r[ok], c[ok]])
        return out

    def clearance(self, xy: FloatArray) -> FloatArray:
        """Distance from each point to the nearest static or moving obstacle surface."""
        r = np.clip(((xy[:, 1] - self.y0) // self.res).astype(int), 0, self.rows - 1)
        c = np.clip(((xy[:, 0] - self.x0) // self.res).astype(int), 0, self.cols - 1)
        d = self.static_dist[r, c]
        if len(self.obstacles):
            dyn = np.min(np.linalg.norm(xy[:, None, :] - self.obstacles[None], axis=-1), axis=1) - self.cfg.obstacle_radius
            d = np.minimum(d, dyn)
        return d

    def _soft_cost(self, d: FloatArray) -> FloatArray:
        c = self.cfg
        band = np.clip((d - c.robot_radius) / c.soft_band, 0.0, 1.0)
        return 1.0 + (c.soft_weight - 1.0) * (1.0 - band)

    def random_free_xy(self, rng: np.random.Generator, min_clearance: float = 0.35) -> tuple[float, float]:
        free = np.argwhere((~self.static_lethal) & (self.static_dist > min_clearance) & self._inside())
        r, c = free[rng.integers(len(free))]
        return self.to_world((int(r), int(c)))

    def _inside(self) -> NDArray[np.bool_]:
        """Cells inside the arena (exclude the map margin beyond the outer walls)."""
        from scipy.ndimage import label  # noqa: PLC0415

        labels, n = label(~self.static_lethal)
        border = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))) - {0}
        sizes = [(int((labels == k).sum()), k) for k in range(1, n + 1) if k not in border]
        return labels == max(sizes)[1] if sizes else ~self.static_lethal
