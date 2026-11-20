"""
Feature extraction front-ends behind one interface (Strategy pattern).

  orb         classical, binary descriptors      (ORB-SLAM's front-end)
  sift        classical, float descriptors
  akaze       classical, binary descriptors
  superpoint  learned CNN (DeTone et al., 2018)  — optional, needs PyTorch + weights

Swapping the extractor is a single ROS parameter (`feature_type`), which is
exactly the knob we turn to decide whether the learned front-end needs a GPU.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .se3 import FloatArray


@dataclass(frozen=True, slots=True)
class Features:
    keypoints: FloatArray  # (N, 2) pixel coordinates (u, v)
    descriptors: NDArray  # (N, D) uint8 (binary) or float32
    binary: bool

    def __len__(self) -> int:
        return int(self.keypoints.shape[0])

    def subset(self, idx: NDArray[np.intp]) -> Features:
        return Features(self.keypoints[idx], self.descriptors[idx], self.binary)

    @classmethod
    def empty(cls, binary: bool, dim: int) -> Features:
        dtype = np.uint8 if binary else np.float32
        return cls(np.zeros((0, 2)), np.zeros((0, dim), dtype=dtype), binary)


class FeatureExtractor(ABC):
    """Gray image -> keypoints + descriptors."""

    name: str = "base"
    binary: bool = False
    descriptor_dim: int = 0

    def __init__(self, max_features: int) -> None:
        self.max_features = max_features
        self.last_ms: float = 0.0

    def extract(self, gray: NDArray[np.uint8]) -> Features:
        t0 = time.perf_counter()
        feats = self._extract(gray)
        self.last_ms = 1000.0 * (time.perf_counter() - t0)
        return feats

    @abstractmethod
    def _extract(self, gray: NDArray[np.uint8]) -> Features: ...

    @property
    def device(self) -> str:
        return "cpu"


class OpenCvExtractor(FeatureExtractor):
    """Shared plumbing for OpenCV's Feature2D detectors."""

    def __init__(self, max_features: int, detector: cv2.Feature2D) -> None:
        super().__init__(max_features)
        self.detector = detector

    def _extract(self, gray: NDArray[np.uint8]) -> Features:
        kps, desc = self.detector.detectAndCompute(gray, None)
        if desc is None or not kps:
            return Features.empty(self.binary, self.descriptor_dim)
        if len(kps) > self.max_features:  # keep the strongest responses
            order = np.argsort([-k.response for k in kps])[: self.max_features]
            kps, desc = [kps[i] for i in order], desc[order]
        uv = np.array([k.pt for k in kps], dtype=np.float64)
        return Features(uv, desc, self.binary)


class OpenCvFactories:
    """Detector constructors differ between OpenCV 4.x and 5.x builds; fail with a clear message."""

    @staticmethod
    def build(name: str, **kwargs: object) -> cv2.Feature2D:
        ctor = getattr(cv2, f"{name}_create", None)
        if ctor is None:
            raise RuntimeError(f"cv2.{name}_create is not available in OpenCV {cv2.__version__}")
        return ctor(**kwargs)


class OrbExtractor(OpenCvExtractor):
    name, binary, descriptor_dim = "orb", True, 32

    def __init__(self, max_features: int = 1000) -> None:
        super().__init__(max_features, OpenCvFactories.build("ORB", nfeatures=max_features, scaleFactor=1.2,
                                                             nlevels=8, fastThreshold=10))


class SiftExtractor(OpenCvExtractor):
    name, binary, descriptor_dim = "sift", False, 128

    def __init__(self, max_features: int = 1000) -> None:
        super().__init__(max_features, OpenCvFactories.build("SIFT", nfeatures=max_features))


class AkazeExtractor(OpenCvExtractor):
    name, binary, descriptor_dim = "akaze", True, 61

    def __init__(self, max_features: int = 1000) -> None:
        super().__init__(max_features, OpenCvFactories.build("AKAZE", threshold=5e-4))


class SuperPointExtractor(FeatureExtractor):
    """
    SuperPoint (Magic Leap). Network definition follows the reference
    `demo_superpoint.py`; weights `superpoint_v1.pth` are loaded from `weights`.

        wget https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth
    """

    name, binary, descriptor_dim = "superpoint", False, 256
    CELL = 8

    def __init__(self, max_features: int = 1000, weights: str = "", device: str = "cpu",
                 conf_thresh: float = 0.015, nms_radius: int = 4) -> None:
        super().__init__(max_features)
        try:
            import torch  # noqa: PLC0415 - optional dependency
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise RuntimeError("feature_type 'superpoint' needs PyTorch: pip install torch") from exc
        if not weights or not Path(weights).is_file():
            raise RuntimeError(f"SuperPoint weights not found at '{weights}' (set superpoint_weights)")
        self.torch = torch
        self._device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.net = _SuperPointNetFactory.build(torch)
        self.net.load_state_dict(torch.load(weights, map_location="cpu"))
        self.net.to(self._device).eval()
        self.conf_thresh = conf_thresh
        self.nms_radius = nms_radius

    @property
    def device(self) -> str:
        return self._device

    def _extract(self, gray: NDArray[np.uint8]) -> Features:
        torch = self.torch
        h, w = gray.shape
        h8, w8 = h - h % self.CELL, w - w % self.CELL
        img = torch.from_numpy(gray[:h8, :w8].astype(np.float32) / 255.0)[None, None].to(self._device)
        with torch.no_grad():
            semi, coarse = self.net(img)
            prob = torch.softmax(semi, dim=1)[:, :-1]  # drop the "no keypoint" dustbin
            hc, wc = h8 // self.CELL, w8 // self.CELL
            heat = prob.permute(0, 2, 3, 1).reshape(1, hc, wc, self.CELL, self.CELL)
            heat = heat.permute(0, 1, 3, 2, 4).reshape(h8, w8)
            # NMS via max-pooling
            pooled = torch.nn.functional.max_pool2d(heat[None, None], 2 * self.nms_radius + 1, 1, self.nms_radius)[0, 0]
            keep = (heat == pooled) & (heat > self.conf_thresh)
            ys, xs = torch.nonzero(keep, as_tuple=True)
            scores = heat[ys, xs]
            order = torch.argsort(scores, descending=True)[: self.max_features]
            ys, xs = ys[order], xs[order]
            # bilinear descriptor sampling
            grid = torch.stack(((xs.float() / (w8 - 1)) * 2 - 1, (ys.float() / (h8 - 1)) * 2 - 1), dim=-1)
            desc = torch.nn.functional.grid_sample(coarse, grid[None, None], align_corners=True)[0, :, 0].T
            desc = torch.nn.functional.normalize(desc, dim=1)
        uv = torch.stack((xs, ys), dim=1).float().cpu().numpy().astype(np.float64)
        return Features(uv, desc.cpu().numpy().astype(np.float32), False)


class _SuperPointNetFactory:
    """Builds the SuperPoint network lazily (keeps torch an optional import)."""

    @staticmethod
    def build(torch):  # noqa: ANN001, ANN205 - torch is a lazily imported module
        nn = torch.nn

        class SuperPointNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
                self.relu = nn.ReLU(inplace=True)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv1a = nn.Conv2d(1, c1, 3, 1, 1)
                self.conv1b = nn.Conv2d(c1, c1, 3, 1, 1)
                self.conv2a = nn.Conv2d(c1, c2, 3, 1, 1)
                self.conv2b = nn.Conv2d(c2, c2, 3, 1, 1)
                self.conv3a = nn.Conv2d(c2, c3, 3, 1, 1)
                self.conv3b = nn.Conv2d(c3, c3, 3, 1, 1)
                self.conv4a = nn.Conv2d(c3, c4, 3, 1, 1)
                self.conv4b = nn.Conv2d(c4, c4, 3, 1, 1)
                self.convPa = nn.Conv2d(c4, c5, 3, 1, 1)
                self.convPb = nn.Conv2d(c5, 65, 1, 1, 0)
                self.convDa = nn.Conv2d(c4, c5, 3, 1, 1)
                self.convDb = nn.Conv2d(c5, d1, 1, 1, 0)

            def forward(self, x):  # noqa: ANN001, ANN202
                r, p = self.relu, self.pool
                x = p(r(self.conv1b(r(self.conv1a(x)))))
                x = p(r(self.conv2b(r(self.conv2a(x)))))
                x = p(r(self.conv3b(r(self.conv3a(x)))))
                x = r(self.conv4b(r(self.conv4a(x))))
                semi = self.convPb(r(self.convPa(x)))
                desc = self.convDb(r(self.convDa(x)))
                return semi, torch.nn.functional.normalize(desc, dim=1)

        return SuperPointNet()


class FeatureExtractorFactory:
    """`feature_type` string -> extractor (Open/Closed: register new front-ends here)."""

    CLASSICAL: dict[str, type[OpenCvExtractor]] = {"orb": OrbExtractor, "sift": SiftExtractor, "akaze": AkazeExtractor}

    @classmethod
    def available(cls) -> list[str]:
        return [*cls.CLASSICAL, "superpoint"]

    @classmethod
    def create(cls, kind: str, max_features: int = 1000, superpoint_weights: str = "",
               device: str = "cpu") -> FeatureExtractor:
        if kind in cls.CLASSICAL:
            return cls.CLASSICAL[kind](max_features)
        if kind == "superpoint":
            return SuperPointExtractor(max_features, superpoint_weights, device)
        raise ValueError(f"unknown feature_type '{kind}', expected one of {cls.available()}")
