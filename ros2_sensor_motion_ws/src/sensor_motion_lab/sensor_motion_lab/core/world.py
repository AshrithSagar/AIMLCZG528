"""
World description — the single source of truth for the environment.

`config/world.yaml` (in mobile_bot_description) is parsed into a `WorldSpec`.
From that one spec we derive:
  * the Gazebo SDF world             (SdfWorldWriter)
  * the occupancy grid / map          (GridRasterizer -> GridMap)
  * ground-truth ray casting          (RayCaster)
so the simulator, the map and the sensor models can never disagree (DRY).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml
from numpy.typing import NDArray

from .geometry import FloatArray


# --------------------------------------------------------------------------- #
# Specification                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned box obstacle (centre + size, metres)."""

    name: str
    cx: float
    cy: float
    sx: float
    sy: float
    height: float = 0.5
    rgba: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax)."""
        return (
            self.cx - self.sx / 2.0,
            self.cy - self.sy / 2.0,
            self.cx + self.sx / 2.0,
            self.cy + self.sy / 2.0,
        )


@dataclass(frozen=True, slots=True)
class WorldSpec:
    """A rectangular arena centred on the origin, with walls and box obstacles."""

    name: str
    size_x: float
    size_y: float
    wall_thickness: float
    wall_height: float
    obstacles: tuple[Box, ...] = field(default_factory=tuple)

    @classmethod
    def from_yaml(cls, path: str | Path) -> WorldSpec:
        with open(path, encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WorldSpec:
        arena = raw["arena"]
        boxes = tuple(
            Box(
                name=str(o["name"]),
                cx=float(o["center"][0]),
                cy=float(o["center"][1]),
                sx=float(o["size"][0]),
                sy=float(o["size"][1]),
                height=float(o.get("height", 0.5)),
                rgba=tuple(float(c) for c in o.get("rgba", (0.6, 0.6, 0.6, 1.0))),  # type: ignore[arg-type]
            )
            for o in raw.get("obstacles", [])
        )
        return cls(
            name=str(raw.get("name", "lab_world")),
            size_x=float(arena["size"][0]),
            size_y=float(arena["size"][1]),
            wall_thickness=float(arena.get("wall_thickness", 0.1)),
            wall_height=float(arena.get("wall_height", 0.5)),
            obstacles=boxes,
        )

    def walls(self) -> tuple[Box, ...]:
        """The four perimeter walls, expressed as boxes."""
        t, hx, hy = self.wall_thickness, self.size_x / 2.0, self.size_y / 2.0
        grey = (0.3, 0.3, 0.3, 1.0)
        return (
            Box("wall_north", 0.0, hy + t / 2, self.size_x + 2 * t, t, self.wall_height, grey),
            Box("wall_south", 0.0, -hy - t / 2, self.size_x + 2 * t, t, self.wall_height, grey),
            Box("wall_east", hx + t / 2, 0.0, t, self.size_y, self.wall_height, grey),
            Box("wall_west", -hx - t / 2, 0.0, t, self.size_y, self.wall_height, grey),
        )

    def all_boxes(self) -> tuple[Box, ...]:
        return self.walls() + self.obstacles


# --------------------------------------------------------------------------- #
# Occupancy grid                                                              #
# --------------------------------------------------------------------------- #
class GridMap:
    """
    Boolean occupancy grid with world<->grid transforms and a lazily
    computed Euclidean distance field (used by the likelihood-field model).
    """

    UNKNOWN_DISTANCE: Final[float] = float("inf")

    def __init__(self, occupied: NDArray[np.bool_], resolution: float, origin_xy: tuple[float, float]) -> None:
        self.occupied: NDArray[np.bool_] = occupied  # indexed [row=gy, col=gx]
        self.resolution: float = resolution
        self.origin_x, self.origin_y = origin_xy
        self.height, self.width = occupied.shape
        self._distance_field: FloatArray | None = None

    # ----- transforms ------------------------------------------------------
    def world_to_grid(self, x: FloatArray, y: FloatArray) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        gx = np.floor((x - self.origin_x) / self.resolution).astype(np.int64)
        gy = np.floor((y - self.origin_y) / self.resolution).astype(np.int64)
        return gx, gy

    def in_bounds(self, gx: NDArray[np.int64], gy: NDArray[np.int64]) -> NDArray[np.bool_]:
        return (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)

    def is_occupied(self, x: FloatArray, y: FloatArray) -> NDArray[np.bool_]:
        """Out-of-map points count as occupied (rays stop at the map edge)."""
        gx, gy = self.world_to_grid(x, y)
        inside = self.in_bounds(gx, gy)
        out = np.ones_like(inside)
        out[inside] = self.occupied[gy[inside], gx[inside]]
        return out

    # ----- distance field --------------------------------------------------
    @property
    def distance_field(self) -> FloatArray:
        """Distance (m) from each cell to the nearest occupied cell (brushfire / BFS)."""
        if self._distance_field is None:
            self._distance_field = self._brushfire()
        return self._distance_field

    def distance_to_nearest(self, x: FloatArray, y: FloatArray, cap: float) -> FloatArray:
        gx, gy = self.world_to_grid(x, y)
        inside = self.in_bounds(gx, gy)
        d = np.full(x.shape, cap, dtype=np.float64)
        d[inside] = np.minimum(self.distance_field[gy[inside], gx[inside]], cap)
        return d

    def _brushfire(self) -> FloatArray:
        """
        Multi-source BFS that propagates the *nearest obstacle cell* outward,
        giving an (almost exact) Euclidean distance transform without SciPy.
        """
        h, w = self.occupied.shape
        nearest = np.full((h, w, 2), -1, dtype=np.int64)
        dist = np.full((h, w), self.UNKNOWN_DISTANCE)
        queue: deque[tuple[int, int]] = deque()
        for gy, gx in zip(*np.nonzero(self.occupied)):
            nearest[gy, gx] = (gy, gx)
            dist[gy, gx] = 0.0
            queue.append((int(gy), int(gx)))
        neighbours = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        while queue:
            cy, cx = queue.popleft()
            oy, ox = nearest[cy, cx]
            for dy, dx in neighbours:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    d = float(np.hypot(ny - oy, nx - ox))
                    if d < dist[ny, nx]:
                        dist[ny, nx] = d
                        nearest[ny, nx] = (oy, ox)
                        queue.append((ny, nx))
        return dist * self.resolution


class GridRasterizer:
    """Turns a `WorldSpec` into a `GridMap`."""

    def __init__(self, resolution: float = 0.05, margin: float = 0.3) -> None:
        self.resolution = resolution
        self.margin = margin

    def rasterize(self, spec: WorldSpec) -> GridMap:
        half_x = spec.size_x / 2.0 + spec.wall_thickness + self.margin
        half_y = spec.size_y / 2.0 + spec.wall_thickness + self.margin
        width = int(np.ceil(2 * half_x / self.resolution))
        height = int(np.ceil(2 * half_y / self.resolution))
        occ = np.zeros((height, width), dtype=bool)
        origin = (-half_x, -half_y)
        for box in spec.all_boxes():
            xmin, ymin, xmax, ymax = box.bounds
            # A cell is occupied iff its *centre* lies inside the box (no systematic inflation).
            gx0 = int(np.ceil((xmin - origin[0]) / self.resolution - 0.5))
            gy0 = int(np.ceil((ymin - origin[1]) / self.resolution - 0.5))
            gx1 = int(np.floor((xmax - origin[0]) / self.resolution - 0.5)) + 1
            gy1 = int(np.floor((ymax - origin[1]) / self.resolution - 0.5)) + 1
            occ[max(gy0, 0) : min(gy1, height), max(gx0, 0) : min(gx1, width)] = True
        return GridMap(occ, self.resolution, origin)


# --------------------------------------------------------------------------- #
# Ground-truth ray casting                                                    #
# --------------------------------------------------------------------------- #
class RayCaster:
    """
    Vectorised ray marching on a `GridMap`.

    Returns the *expected* range z* for every (pose, beam) pair — the quantity
    the beam model needs and the simulator uses to synthesise scans.
    """

    def __init__(self, grid: GridMap, max_range: float, step: float | None = None) -> None:
        self.grid = grid
        self.max_range = max_range
        self.step = step if step is not None else grid.resolution / 2.0
        self._ts: FloatArray = np.arange(0.0, max_range + self.step, self.step)

    def cast(self, poses: FloatArray, beam_angles: FloatArray) -> FloatArray:
        """
        poses: (N, 3) sensor poses in the map frame; beam_angles: (B,) in the sensor frame.
        Returns (N, B) ranges, clipped to `max_range`.
        """
        poses = np.atleast_2d(poses)
        heading = poses[:, 2:3] + beam_angles[None, :]  # (N, B)
        cos_h, sin_h = np.cos(heading)[..., None], np.sin(heading)[..., None]
        xs = poses[:, 0, None, None] + cos_h * self._ts  # (N, B, T)
        ys = poses[:, 1, None, None] + sin_h * self._ts
        hit = self.grid.is_occupied(xs, ys)
        any_hit = hit.any(axis=-1)
        first = np.argmax(hit, axis=-1)
        # The surface lies between the last free and the first occupied sample: take the midpoint.
        t_hit = np.maximum(self._ts[first] - self.step / 2.0, 0.0)
        ranges = np.where(any_hit, t_hit, self.max_range)
        return np.minimum(ranges, self.max_range)


# --------------------------------------------------------------------------- #
# Gazebo export                                                               #
# --------------------------------------------------------------------------- #
class SdfWorldWriter:
    """Renders a `WorldSpec` as a Gazebo (gz-sim Harmonic) SDF world."""

    SYSTEM_PLUGINS: Final[tuple[tuple[str, str, str], ...]] = (
        ("gz-sim-physics-system", "gz::sim::systems::Physics", ""),
        ("gz-sim-user-commands-system", "gz::sim::systems::UserCommands", ""),
        ("gz-sim-scene-broadcaster-system", "gz::sim::systems::SceneBroadcaster", ""),
        ("gz-sim-sensors-system", "gz::sim::systems::Sensors", "<render_engine>ogre2</render_engine>"),
    )

    def __init__(self, physics_step: float = 0.004, real_time_factor: float = 1.0) -> None:
        self.physics_step = physics_step
        self.real_time_factor = real_time_factor

    def render(self, spec: WorldSpec) -> str:
        plugins = "\n".join(
            f'    <plugin filename="{fn}" name="{nm}">{extra}</plugin>' for fn, nm, extra in self.SYSTEM_PLUGINS
        )
        models = "\n".join(self._box_model(b) for b in spec.all_boxes())
        return f"""<?xml version="1.0" ?>
<!-- AUTO-GENERATED from world.yaml by SdfWorldWriter. Do not edit by hand. -->
<sdf version="1.9">
  <world name="{spec.name}">
    <physics name="default_physics" type="ode">
      <max_step_size>{self.physics_step}</max_step_size>
      <real_time_factor>{self.real_time_factor}</real_time_factor>
    </physics>
{plugins}
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>50 50</size></plane></geometry></collision>
        <visual name="visual"><geometry><plane><normal>0 0 1</normal><size>50 50</size></plane></geometry>
          <material><ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse></material></visual>
      </link>
    </model>
{models}
  </world>
</sdf>
"""

    @staticmethod
    def _box_model(b: Box) -> str:
        rgba = " ".join(f"{c:g}" for c in b.rgba)
        geom = f"<geometry><box><size>{b.sx} {b.sy} {b.height}</size></box></geometry>"
        return f"""    <model name="{b.name}">
      <static>true</static>
      <pose>{b.cx} {b.cy} {b.height / 2.0} 0 0 0</pose>
      <link name="link">
        <collision name="collision">{geom}</collision>
        <visual name="visual">{geom}<material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material></visual>
      </link>
    </model>"""

    def write(self, spec: WorldSpec, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(self.render(spec), encoding="utf-8")
        return out
