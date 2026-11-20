"""
CPU-only synthetic RGB-D source with exact ground truth.

A textured room (inverted box) with a few textured boxes inside, rendered by
vectorised ray casting. No GPU, no OpenGL, no dataset download — the same
scene description also generates the Gazebo world (see `GazeboWorldWriter`),
so the two sources are directly comparable.

World frame: z up. Camera frames: REP-103 optical (z forward, x right, y down).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .camera import PinholeCamera, RgbdFrame
from .se3 import SE3, SO3, FloatArray


# --------------------------------------------------------------------------- #
# Scene description                                                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TexturedBox:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    texture: int  # index into TextureBank
    inverted: bool = False  # True for the room: we see the inside faces

    @property
    def lo(self) -> FloatArray:
        return np.asarray(self.center) - np.asarray(self.size) / 2.0

    @property
    def hi(self) -> FloatArray:
        return np.asarray(self.center) + np.asarray(self.size) / 2.0


@dataclass(frozen=True, slots=True)
class SceneSpec:
    boxes: tuple[TexturedBox, ...]
    texture_metres: float = 8.0  # one texture tile covers this many metres (> any face: no repetition)

    @classmethod
    def default_room(cls) -> SceneSpec:
        return cls(
            (
                TexturedBox("room", (0.0, 0.0, 1.5), (8.0, 8.0, 3.0), 0, inverted=True),
                TexturedBox("crate_a", (2.2, 1.5, 0.5), (0.8, 0.8, 1.0), 1),
                TexturedBox("crate_b", (-2.0, -2.2, 0.6), (1.0, 0.6, 1.2), 2),
                TexturedBox("pillar", (-1.8, 2.3, 1.5), (0.4, 0.4, 3.0), 3),
                TexturedBox("shelf", (2.6, -2.0, 0.9), (0.5, 1.6, 1.8), 4),
            )
        )


class TextureBank:
    """Deterministic procedural textures: blobs + rectangles + text-like strokes (corner-rich)."""

    def __init__(self, count: int = 16, size: int = 1024, seed: int = 7) -> None:
        rng = np.random.default_rng(seed)
        self.size = size
        self.textures: list[NDArray[np.uint8]] = [self._make(rng) for _ in range(count)]

    def _make(self, rng: np.random.Generator) -> NDArray[np.uint8]:
        s = self.size
        base = rng.integers(60, 200, 3)
        img = np.empty((s, s, 3), np.uint8)
        img[:] = base
        noise = cv2.GaussianBlur(rng.normal(0, 40, (s, s, 3)).astype(np.float32), (0, 0), 6)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        for _ in range(160):
            c = tuple(int(v) for v in rng.integers(0, 256, 3))
            x0, y0 = (int(v) for v in rng.integers(0, s, 2))
            w, h = (int(v) for v in rng.integers(15, 120, 2))
            cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), c, -1 if rng.random() < 0.6 else 3)
        for _ in range(100):
            c = tuple(int(v) for v in rng.integers(0, 256, 3))
            cv2.circle(img, tuple(int(v) for v in rng.integers(0, s, 2)), int(rng.integers(5, 40)), c, -1)
        for _ in range(48):
            c = tuple(int(v) for v in rng.integers(0, 256, 3))
            org = tuple(int(v) for v in rng.integers(0, s - 40, 2))
            cv2.putText(img, "".join(rng.choice(list("ABCDEFXYZ0123456789"), 4)), org,
                        cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.6, 1.4)), c, 2)
        return img

    def write_png(self, folder: Path) -> list[Path]:
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, t in enumerate(self.textures):
            p = folder / f"texture_{i}.png"
            cv2.imwrite(str(p), cv2.cvtColor(t, cv2.COLOR_RGB2BGR))
            paths.append(p)
        return paths


# --------------------------------------------------------------------------- #
# Renderer                                                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SensorNoise:
    depth_std_at_1m: float = 0.0015  # Kinect-like: sigma grows with z^2
    rgb_std: float = 2.0
    seed: int = 0


class RayCastRenderer:
    """Renders colour + metric depth for a camera pose by slab-method ray/box intersection."""

    def __init__(self, scene: SceneSpec, camera: PinholeCamera, textures: TextureBank | None = None,
                 noise: SensorNoise = SensorNoise(), max_depth: float = 10.0) -> None:
        self.scene = scene
        self.camera = camera
        self.textures = textures or TextureBank()
        self.noise = noise
        self.max_depth = max_depth
        rays = camera.ray_directions()
        self._rays_cam = rays.reshape(-1, 3)
        self._z_per_t = 1.0 / np.linalg.norm(self._rays_cam, axis=1)  # depth = t * z-component of unit ray
        self._rays_unit = self._rays_cam * self._z_per_t[:, None]

    def render(self, T_world_cam: SE3, stamp: float = 0.0, index: int = 0) -> RgbdFrame:
        """`index` seeds the sensor noise, so re-rendering a frame is reproducible."""
        rng = np.random.default_rng((self.noise.seed, index))
        h, w = self.camera.height, self.camera.width
        d = self._rays_unit @ T_world_cam.rotation.T  # (P, 3) unit directions in world
        o = T_world_cam.translation
        best_t = np.full(d.shape[0], np.inf)
        rgb = np.zeros((d.shape[0], 3), np.uint8)
        for box in self.scene.boxes:
            t, axis, side = self._intersect(o, d, box)
            closer = t < best_t
            if not closer.any():
                continue
            best_t[closer] = t[closer]
            p = o + d[closer] * t[closer, None]
            rgb[closer] = self._shade(p, axis[closer], side[closer], box)
        depth = best_t * self._z_per_t
        depth[~np.isfinite(depth) | (depth > self.max_depth)] = 0.0
        depth = self._noisy_depth(depth, rng)
        img = rgb.reshape(h, w, 3).astype(np.float32)
        if self.noise.rgb_std > 0:
            img += rng.normal(0, self.noise.rgb_std, img.shape)
        return RgbdFrame(stamp, np.clip(img, 0, 255).astype(np.uint8), depth.reshape(h, w).astype(np.float32))

    # ----- internals ----------------------------------------------------------
    @staticmethod
    def _intersect(o: FloatArray, d: FloatArray, box: TexturedBox) -> tuple[FloatArray, NDArray, NDArray]:
        """Hit distance (inf if none), hit axis (0/1/2) and side (0 = low face, 1 = high face)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            t1 = (box.lo - o) * inv
            t2 = (box.hi - o) * inv
        tmin_ax, tmax_ax = np.minimum(t1, t2), np.maximum(t1, t2)
        tmin_ax = np.nan_to_num(tmin_ax, nan=-np.inf)
        tmax_ax = np.nan_to_num(tmax_ax, nan=np.inf)
        t_near, t_far = tmin_ax.max(axis=1), tmax_ax.min(axis=1)
        hit = (t_far >= np.maximum(t_near, 0.0))
        if box.inverted:  # inside the room: exit point
            t = np.where(hit, t_far, np.inf)
            axis = tmax_ax.argmin(axis=1)
            side = (d[np.arange(len(d)), axis] > 0).astype(int)
        else:
            t = np.where(hit & (t_near > 1e-6), t_near, np.inf)
            axis = tmin_ax.argmax(axis=1)
            side = (d[np.arange(len(d)), axis] < 0).astype(int)
        return t, axis, side

    def _shade(self, p: FloatArray, axis: NDArray, side: NDArray, box: TexturedBox) -> NDArray[np.uint8]:
        bank = self.textures.textures
        s = bank[0].shape[0]
        # in-plane coordinates of each face (the two axes other than `axis`)
        uv = np.empty((p.shape[0], 2))
        for ax in range(3):
            m = axis == ax
            others = [a for a in range(3) if a != ax]
            uv[m] = p[m][:, others]
        k = np.floor(uv / self.scene.texture_metres * s).astype(int) % s
        face = axis * 2 + side  # 0..5: every face of every box gets its own texture
        col = np.empty((p.shape[0], 3), np.float32)
        for f in np.unique(face):
            m = face == f
            tex = bank[(6 * box.texture + int(f)) % len(bank)]
            col[m] = tex[k[m, 1], k[m, 0]]
        light = 0.75 + 0.25 * np.where(axis == 2, 1.0, 0.6 + 0.4 * side)  # faces get slightly different shading
        return np.clip(col * light[:, None], 0, 255).astype(np.uint8)

    def _noisy_depth(self, depth: FloatArray, rng: np.random.Generator) -> FloatArray:
        if self.noise.depth_std_at_1m <= 0:
            return depth
        valid = depth > 0
        out = depth.copy()
        out[valid] += rng.normal(0, 1, valid.sum()) * self.noise.depth_std_at_1m * depth[valid] ** 2
        return out


# --------------------------------------------------------------------------- #
# Trajectory                                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class LoopTrajectory:
    """
    A closed loop: the camera circles the room centre at `radius`, height `z`,
    looking outward-and-forward (`yaw_offset` from the tangent), with a gentle
    bob in height and pitch — so the sequence ends where it started (loop closure!).
    """

    radius: float = 1.3
    z: float = 1.3
    period: float = 40.0  # seconds per loop
    loops: float = 1.1
    yaw_offset_deg: float = 35.0
    bob: float = 0.10
    rate_hz: float = 15.0

    @property
    def duration(self) -> float:
        return self.period * self.loops

    def stamps(self) -> FloatArray:
        return np.arange(0.0, self.duration, 1.0 / self.rate_hz)

    def pose(self, t: float) -> SE3:
        a = 2 * math.pi * t / self.period
        pos = np.array([self.radius * math.cos(a), self.radius * math.sin(a),
                        self.z + self.bob * math.sin(3 * a)])
        heading = a + math.pi / 2 - math.radians(self.yaw_offset_deg)  # tangent, turned outward
        pitch = math.radians(5.0) * math.sin(2 * a)
        # body (x fwd, y left, z up) -> optical (z fwd, x right, y down)
        r_world_body = SO3.exp(np.array([0, 0, heading])) @ SO3.exp(np.array([0, pitch, 0]))
        r_body_opt = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
        return SE3(r_world_body @ r_body_opt, pos)


@dataclass
class SyntheticSequence:
    """Iterable RGB-D sequence with ground truth (the 'dataset' interface)."""

    camera: PinholeCamera = field(default_factory=lambda: PinholeCamera.tum_default().scaled(0.5))
    trajectory: LoopTrajectory = field(default_factory=LoopTrajectory)
    scene: SceneSpec = field(default_factory=SceneSpec.default_room)
    noise: SensorNoise = field(default_factory=SensorNoise)
    name: str = "synthetic_room_loop"

    def __post_init__(self) -> None:
        self.renderer = RayCastRenderer(self.scene, self.camera, noise=self.noise)
        self.stamps = self.trajectory.stamps()

    def __len__(self) -> int:
        return int(self.stamps.size)

    def ground_truth(self, i: int) -> SE3:
        return self.trajectory.pose(float(self.stamps[i]))

    def frame(self, i: int) -> RgbdFrame:
        return self.renderer.render(self.ground_truth(i), float(self.stamps[i]), i)
