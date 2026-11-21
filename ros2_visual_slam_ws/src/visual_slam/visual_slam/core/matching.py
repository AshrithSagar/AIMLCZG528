"""
Feature matching behind one interface (Strategy pattern).

  bruteforce  k-NN + Lowe's ratio test + mutual check      (works for every front-end)
  lightglue   learned attention-based matcher               (Lindenberger, Sarlin, Pollefeys, ICCV 2023)
              — needs `superpoint` features, PyTorch and the `lightglue` package

The matcher is used twice in the pipeline: frame-to-keyframe tracking and the
geometric check of a loop candidate. Both see the same `match()` call, so the
learned matcher improves both without either caller changing.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import ClassVar

import cv2
import numpy as np
from numpy.typing import NDArray

from .features import Features
from .runtime import TorchRuntime


class FeatureMatcher(ABC):
    name: ClassVar[str] = "base"

    def __init__(self) -> None:
        self.last_ms = 0.0

    def match(self, query: Features, train: Features) -> NDArray[np.intp]:
        """(M, 2) index pairs (query_idx, train_idx)."""
        if len(query) < 2 or len(train) < 2:
            return np.zeros((0, 2), dtype=np.intp)
        t0 = time.perf_counter()
        pairs = self._match(query, train)
        self.last_ms = 1000.0 * (time.perf_counter() - t0)
        return np.asarray(pairs, dtype=np.intp).reshape(-1, 2)

    @abstractmethod
    def _match(self, query: Features, train: Features) -> NDArray[np.intp]: ...

    @property
    def device(self) -> str:
        return "cpu"


class DescriptorMatcher(FeatureMatcher):
    """Brute-force k-NN with Lowe's ratio test and a mutual (cross) check."""

    name = "bruteforce"

    def __init__(self, ratio: float = 0.8) -> None:
        super().__init__()
        self.ratio = ratio
        self._bf_cache: dict[bool, cv2.BFMatcher] = {}

    def _bf(self, binary: bool) -> cv2.BFMatcher:
        if binary not in self._bf_cache:
            self._bf_cache[binary] = cv2.BFMatcher(cv2.NORM_HAMMING if binary else cv2.NORM_L2)
        return self._bf_cache[binary]

    def _match(self, query: Features, train: Features) -> NDArray[np.intp]:
        bf = self._bf(query.binary)
        fwd = bf.knnMatch(query.descriptors, train.descriptors, k=2)
        good = {m.queryIdx: m.trainIdx for m, n in (p for p in fwd if len(p) == 2) if m.distance < self.ratio * n.distance}
        if not good:
            return np.zeros((0, 2), dtype=np.intp)
        back = {m.queryIdx: m.trainIdx for m in bf.match(train.descriptors, query.descriptors)}
        return np.asarray([(q, t) for q, t in good.items() if back.get(t) == q], dtype=np.intp)


class LightGlueMatcher(FeatureMatcher):
    """
    LightGlue: a transformer that attends over *both* keypoint sets (positions +
    descriptors) and outputs a partial assignment. It replaces the hand-tuned
    ratio test, and it stops early on easy pairs (adaptive depth/width).
    """

    name = "lightglue"
    SUPPORTED: ClassVar[tuple[str, ...]] = ("superpoint",)

    def __init__(self, features: str = "superpoint", device: str = "cpu", min_score: float = 0.1) -> None:
        super().__init__()
        if features not in self.SUPPORTED:
            raise ValueError(f"matcher 'lightglue' needs learned features {self.SUPPORTED}, not '{features}'")
        self.rt = TorchRuntime(device)
        lightglue = TorchRuntime.require("lightglue")
        self.net = lightglue.LightGlue(features=features, filter_threshold=min_score).eval().to(self.rt.device)

    @property
    def device(self) -> str:
        return self.rt.device

    def _as_input(self, f: Features) -> dict:
        t = self.rt.torch
        return {
            "keypoints": self.rt.tensor(f.keypoints.astype(np.float32))[None],
            "descriptors": self.rt.tensor(np.ascontiguousarray(f.descriptors, dtype=np.float32))[None],
            "image_size": t.tensor([f.image_size], dtype=t.float32, device=self.rt.device),
        }

    def _match(self, query: Features, train: Features) -> NDArray[np.intp]:
        if query.binary or train.binary:
            raise ValueError("LightGlue needs float (learned) descriptors")
        with self.rt.no_grad():
            out = self.net({"image0": self._as_input(query), "image1": self._as_input(train)})
        self.rt.synchronize()
        return out["matches"][0].cpu().numpy().astype(np.intp)  # (K, 2): image0 = query, image1 = train


class MatcherFactory:
    NAMES: ClassVar[tuple[str, ...]] = ("bruteforce", "lightglue")

    @classmethod
    def create(cls, kind: str, feature_type: str, ratio: float = 0.8, device: str = "cpu") -> FeatureMatcher:
        if kind == "bruteforce":
            return DescriptorMatcher(ratio)
        if kind == "lightglue":
            return LightGlueMatcher(feature_type, device)
        raise ValueError(f"unknown matcher '{kind}', expected one of {cls.NAMES}")
