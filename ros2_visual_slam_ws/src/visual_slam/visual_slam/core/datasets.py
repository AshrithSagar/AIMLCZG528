"""
RGB-D sequence sources behind one interface:

  TumSequence        TUM RGB-D benchmark folder (rgb.txt, depth.txt, groundtruth.txt)
  SyntheticSequence  CPU ray-cast room (see synthetic.py)

Plus `TumExporter`, which writes any sequence in TUM format — so the synthetic
loop can be fed to `evo`, RTAB-Map or ORB-SLAM exactly like a real TUM sequence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from .camera import PinholeCamera, RgbdFrame
from .se3 import SE3, FloatArray
from .synthetic import SyntheticSequence


class RgbdSequence(ABC):
    name: str
    camera: PinholeCamera

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def frame(self, i: int) -> RgbdFrame: ...

    @abstractmethod
    def ground_truth(self, i: int) -> SE3 | None: ...

    def stamp(self, i: int) -> float:
        return self.frame(i).stamp


class SyntheticSource(RgbdSequence):
    """Adapter giving `SyntheticSequence` the common interface."""

    def __init__(self, seq: SyntheticSequence | None = None) -> None:
        self.seq = seq or SyntheticSequence()
        self.name, self.camera = self.seq.name, self.seq.camera

    def __len__(self) -> int:
        return len(self.seq)

    def frame(self, i: int) -> RgbdFrame:
        return self.seq.frame(i)

    def ground_truth(self, i: int) -> SE3:
        return self.seq.ground_truth(i)

    def stamp(self, i: int) -> float:
        return float(self.seq.stamps[i])


class TumAssociation:
    """Nearest-timestamp association (as in the TUM `associate.py` tool)."""

    @staticmethod
    def read_list(path: Path) -> list[tuple[float, list[str]]]:
        rows = []
        for line in path.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                parts = line.split()
                rows.append((float(parts[0]), parts[1:]))
        return rows

    @staticmethod
    def associate(a: FloatArray, b: FloatArray, max_dt: float) -> list[tuple[int, int]]:
        pairs = []
        for i, t in enumerate(a):
            k = int(np.argmin(np.abs(b - t)))
            if abs(b[k] - t) <= max_dt:
                pairs.append((i, k))
        return pairs


class TumSequence(RgbdSequence):
    """
    A TUM RGB-D sequence folder, e.g. `rgbd_dataset_freiburg1_xyz/`
    (https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download).
    Depth PNGs are uint16 with a factor of 5000 per metre.
    """

    DEPTH_FACTOR = 5000.0

    def __init__(self, folder: str | Path, camera: PinholeCamera | None = None, max_dt: float = 0.02) -> None:
        self.root = Path(folder).expanduser()
        self.name = self.root.name
        self.camera = camera or self._guess_camera()
        rgb = TumAssociation.read_list(self.root / "rgb.txt")
        dep = TumAssociation.read_list(self.root / "depth.txt")
        pairs = TumAssociation.associate(np.array([r[0] for r in rgb]), np.array([d[0] for d in dep]), max_dt)
        self.items = [(rgb[i][0], rgb[i][1][0], dep[k][1][0]) for i, k in pairs]
        self.gt_stamps, self.gt_poses = self._read_groundtruth()

    def _guess_camera(self) -> PinholeCamera:
        if "freiburg1" in self.name:
            return PinholeCamera.tum_freiburg1()
        return PinholeCamera.tum_default()

    def _read_groundtruth(self) -> tuple[FloatArray, list[SE3]]:
        path = self.root / "groundtruth.txt"
        if not path.is_file():
            return np.zeros(0), []
        rows = TumAssociation.read_list(path)
        stamps = np.array([r[0] for r in rows])
        poses = [SE3.from_quaternion(np.array(r[1][3:7], float), np.array(r[1][0:3], float)) for r in rows]
        return stamps, poses

    def __len__(self) -> int:
        return len(self.items)

    def stamp(self, i: int) -> float:
        return self.items[i][0]

    def frame(self, i: int) -> RgbdFrame:
        t, rgb_rel, depth_rel = self.items[i]
        bgr = cv2.imread(str(self.root / rgb_rel), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(self.root / depth_rel), cv2.IMREAD_UNCHANGED).astype(np.float32) / self.DEPTH_FACTOR
        return RgbdFrame(t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth)

    def ground_truth(self, i: int) -> SE3 | None:
        if not self.gt_poses:
            return None
        k = int(np.argmin(np.abs(self.gt_stamps - self.stamp(i))))
        return self.gt_poses[k] if abs(self.gt_stamps[k] - self.stamp(i)) < 0.02 else None


class TumExporter:
    """Writes a sequence to disk in TUM RGB-D layout."""

    @staticmethod
    def write(seq: RgbdSequence, folder: str | Path, every: int = 1) -> Path:
        root = Path(folder).expanduser()
        (root / "rgb").mkdir(parents=True, exist_ok=True)
        (root / "depth").mkdir(parents=True, exist_ok=True)
        rgb_lines, dep_lines, gt_lines = ["# color images"], ["# depth maps"], ["# timestamp tx ty tz qx qy qz qw"]
        for i in range(0, len(seq), every):
            f = seq.frame(i)
            name = f"{f.stamp:.6f}.png"
            cv2.imwrite(str(root / "rgb" / name), cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(root / "depth" / name), np.round(f.depth * TumSequence.DEPTH_FACTOR).astype(np.uint16))
            rgb_lines.append(f"{f.stamp:.6f} rgb/{name}")
            dep_lines.append(f"{f.stamp:.6f} depth/{name}")
            gt = seq.ground_truth(i)
            if gt is not None:
                q = gt.quaternion()
                gt_lines.append(f"{f.stamp:.6f} " + " ".join(f"{v:.6f}" for v in (*gt.translation, *q)))
        (root / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
        (root / "depth.txt").write_text("\n".join(dep_lines) + "\n")
        (root / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n")
        c = seq.camera
        (root / "camera.txt").write_text(f"fx fy cx cy width height\n{c.fx} {c.fy} {c.cx} {c.cy} {c.width} {c.height}\n")
        return root


class SequenceFactory:
    @staticmethod
    def create(source: str, path: str = "") -> RgbdSequence:
        if source == "synthetic":
            return SyntheticSource()
        if source == "tum":
            folder = Path(path).expanduser()
            cam_file = folder / "camera.txt"  # written by TumExporter (synthetic exported to disk)
            cam = None
            if cam_file.is_file():
                fx, fy, cx, cy, w, h = cam_file.read_text().splitlines()[1].split()
                cam = PinholeCamera(float(fx), float(fy), float(cx), float(cy), int(float(w)), int(float(h)))
            return TumSequence(folder, cam)
        raise ValueError(f"unknown source '{source}' (expected 'synthetic' or 'tum')")
