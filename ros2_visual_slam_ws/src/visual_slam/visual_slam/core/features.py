"""
Feature extraction front-ends behind one interface (Strategy pattern).

  orb         classical, binary descriptors      (ORB-SLAM's front-end)
  sift        classical, float descriptors
  akaze       classical, binary descriptors
  superpoint  learned CNN (DeTone et al., 2018)  — optional: PyTorch + the `lightglue` package
              (weights download automatically on first use)

Swapping the extractor is a single ROS parameter (`feature_type`), which is
exactly the knob we turn to decide whether the learned front-end needs a GPU.
SuperPoint descriptors can be matched by brute force (like ORB) or by the
learned LightGlue matcher (see matching.py).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from .runtime import TorchRuntime
from .se3 import FloatArray


@dataclass(frozen=True, slots=True)
class Features:
    keypoints: FloatArray  # (N, 2) pixel coordinates (u, v)
    descriptors: NDArray  # (N, D) uint8 (binary) or float32
    binary: bool
    image_size: tuple[int, int] = (0, 0)  # (width, height): learned matchers normalise keypoints by it

    def __len__(self) -> int:
        return int(self.keypoints.shape[0])

    def subset(self, idx: NDArray[np.intp]) -> Features:
        return Features(self.keypoints[idx], self.descriptors[idx], self.binary, self.image_size)

    @classmethod
    def empty(cls, binary: bool, dim: int, image_size: tuple[int, int] = (0, 0)) -> Features:
        dtype = np.uint8 if binary else np.float32
        return cls(np.zeros((0, 2)), np.zeros((0, dim), dtype=dtype), binary, image_size)


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

    @staticmethod
    def size_of(gray: NDArray[np.uint8]) -> tuple[int, int]:
        return int(gray.shape[1]), int(gray.shape[0])

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
            return Features.empty(self.binary, self.descriptor_dim, self.size_of(gray))
        if len(kps) > self.max_features:  # keep the strongest responses
            order = np.argsort([-k.response for k in kps])[: self.max_features]
            kps, desc = [kps[i] for i in order], desc[order]
        uv = np.array([k.pt for k in kps], dtype=np.float64)
        return Features(uv, desc, self.binary, self.size_of(gray))


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
    SuperPoint (DeTone, Malisiewicz, Rabinovich, CVPRW 2018) via the `lightglue`
    package, which ships the network and downloads the Magic Leap weights on first use.
    Descriptors are 256-D, L2-normalised floats: they work with brute-force L2
    matching *and* with LightGlue.
    """

    name, binary, descriptor_dim = "superpoint", False, 256

    def __init__(self, max_features: int = 1000, device: str = "cpu", detection_threshold: float = 0.0005,
                 nms_radius: int = 4) -> None:
        super().__init__(max_features)
        self.rt = TorchRuntime(device)
        lightglue = TorchRuntime.require("lightglue")
        self.net = lightglue.SuperPoint(max_num_keypoints=max_features, detection_threshold=detection_threshold,
                                        nms_radius=nms_radius).eval().to(self.rt.device)

    @property
    def device(self) -> str:
        return self.rt.device

    def _extract(self, gray: NDArray[np.uint8]) -> Features:
        img = self.rt.tensor(np.ascontiguousarray(gray, dtype=np.float32) / 255.0)[None]  # (1, H, W) in [0, 1]
        with self.rt.no_grad():
            out = self.net.extract(img, resize=None)  # keep the native resolution
        self.rt.synchronize()
        uv = out["keypoints"][0].cpu().numpy().astype(np.float64)
        desc = out["descriptors"][0].cpu().numpy().astype(np.float32)
        return Features(uv, desc, False, self.size_of(gray))


class FeatureExtractorFactory:
    """`feature_type` string -> extractor (Open/Closed: register new front-ends here)."""

    CLASSICAL: dict[str, type[OpenCvExtractor]] = {"orb": OrbExtractor, "sift": SiftExtractor, "akaze": AkazeExtractor}

    LEARNED = ("superpoint",)

    @classmethod
    def available(cls) -> list[str]:
        return [*cls.CLASSICAL, *cls.LEARNED]

    @classmethod
    def create(cls, kind: str, max_features: int = 1000, device: str = "cpu") -> FeatureExtractor:
        if kind in cls.CLASSICAL:
            return cls.CLASSICAL[kind](max_features)
        if kind == "superpoint":
            return SuperPointExtractor(max_features, device)
        raise ValueError(f"unknown feature_type '{kind}', expected one of {cls.available()}")
