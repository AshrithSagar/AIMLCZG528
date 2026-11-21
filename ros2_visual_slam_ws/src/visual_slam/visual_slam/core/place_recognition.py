"""
Place recognition: "have I been here before?" — the first half of loop closure.
The second half (geometric verification with PnP) is shared, in backend.py.

  proximity  only look at old keyframes near the *estimated* position, score them by
             local-descriptor voting                       (our original; needs the pose prior)
  thumbnail  global descriptor = tiny normalised image     (classical, SeqSLAM-style; no pose prior)
  dinov2     global descriptor from a self-supervised ViT  (learned; Oquab et al. 2023, as in AnyLoc)

The retrieval variants search *all* old keyframes by appearance alone, so they
still find a revisit when odometry has drifted too far for a pose prior to help.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import cv2
import numpy as np
from numpy.typing import NDArray

from .matching import FeatureMatcher
from .runtime import TorchRuntime
from .se3 import FloatArray
from .tracking import Keyframe


@dataclass(frozen=True, slots=True)
class PlaceCandidate:
    keyframe: Keyframe
    score: float
    pairs: NDArray[np.intp] | None = None  # local matches already computed while scoring (reused by verification)


# --------------------------------------------------------------------------- #
# Global image descriptors                                                    #
# --------------------------------------------------------------------------- #
class GlobalDescriptor(ABC):
    """RGB image -> one L2-normalised vector; similar places -> high cosine similarity."""

    name: ClassVar[str] = "base"
    min_similarity: ClassVar[float] = 0.0  # below this a candidate is not even verified

    def __init__(self) -> None:
        self.last_ms = 0.0

    def describe(self, rgb: NDArray[np.uint8]) -> FloatArray:
        t0 = time.perf_counter()
        v = np.asarray(self._describe(rgb), dtype=np.float32).ravel()
        self.last_ms = 1000.0 * (time.perf_counter() - t0)
        return v / max(float(np.linalg.norm(v)), 1e-12)

    @abstractmethod
    def _describe(self, rgb: NDArray[np.uint8]) -> FloatArray: ...

    @property
    def device(self) -> str:
        return "cpu"


class ThumbnailDescriptor(GlobalDescriptor):
    """Downsample to a few hundred pixels and normalise contrast patch-wise (Milford & Wyeth, SeqSLAM 2012)."""

    name = "thumbnail"
    min_similarity = 0.3

    def __init__(self, size: tuple[int, int] = (32, 24), patch: int = 8) -> None:
        super().__init__()
        self.size, self.patch = size, patch

    def _describe(self, rgb: NDArray[np.uint8]) -> FloatArray:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA).astype(np.float32)
        p = self.patch
        h, w = small.shape
        blocks = small[: h - h % p, : w - w % p].reshape(h // p, p, w // p, p).swapaxes(1, 2)
        mean = blocks.mean(axis=(2, 3), keepdims=True)
        std = blocks.std(axis=(2, 3), keepdims=True) + 1e-3
        return ((blocks - mean) / std).ravel()


class Dinov2Descriptor(GlobalDescriptor):
    """
    DINOv2 ViT (self-supervised, never trained for place recognition) used as a
    frozen feature extractor: [CLS token ; mean of patch tokens], L2-normalised.
    Computed once per *keyframe*, not per frame, so it is affordable on a CPU.
    """

    name = "dinov2"
    min_similarity = 0.5

    def __init__(self, model_id: str = "facebook/dinov2-small", device: str = "cpu") -> None:
        super().__init__()
        self.rt = TorchRuntime(device)
        transformers = TorchRuntime.require("transformers")
        self.processor = transformers.AutoImageProcessor.from_pretrained(model_id)
        self.net = transformers.AutoModel.from_pretrained(model_id).eval().to(self.rt.device)

    @property
    def device(self) -> str:
        return self.rt.device

    def _describe(self, rgb: NDArray[np.uint8]) -> FloatArray:
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.rt.device)
        with self.rt.no_grad():
            tokens = self.net(**inputs).last_hidden_state[0]  # (1 + patches, D)
        torch = self.rt.torch
        cls_tok = torch.nn.functional.normalize(tokens[0], dim=0)
        patches = torch.nn.functional.normalize(tokens[1:].mean(dim=0), dim=0)
        return torch.cat((cls_tok, patches)).cpu().numpy()


class GlobalDescriptorFactory:
    @staticmethod
    def create(kind: str, device: str = "cpu") -> GlobalDescriptor:
        if kind == "thumbnail":
            return ThumbnailDescriptor()
        if kind == "dinov2":
            return Dinov2Descriptor(device=device)
        raise ValueError(f"unknown global descriptor '{kind}' (thumbnail | dinov2)")


# --------------------------------------------------------------------------- #
# Recognisers: keyframe database + query                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RecognitionConfig:
    min_separation: int = 30  # only consider keyframes at least this many keyframes old
    search_radius: float = 2.0  # m (proximity only), around the drifting odometry estimate
    max_checked: int = 15  # proximity only: nearest old keyframes scored by voting
    min_vote_score: float = 0.08  # proximity only: matched fraction of the query's features


class PlaceRecognizer(ABC):
    name: ClassVar[str] = "base"

    def __init__(self, config: RecognitionConfig) -> None:
        self.config = config
        self.last_ms = 0.0

    def add(self, kf: Keyframe) -> None:
        """Insert a keyframe into the database (called for *every* keyframe)."""

    def query(self, kf: Keyframe, keyframes: list[Keyframe]) -> list[PlaceCandidate]:
        """Old keyframes that may show the same place, best first."""
        t0 = time.perf_counter()
        old = [k for k in keyframes if k.id <= kf.id - self.config.min_separation]
        out = self._query(kf, old) if old else []
        self.last_ms = 1000.0 * (time.perf_counter() - t0)
        return out

    @abstractmethod
    def _query(self, kf: Keyframe, old: list[Keyframe]) -> list[PlaceCandidate]: ...

    @property
    def device(self) -> str:
        return "cpu"


class ProximityVoting(PlaceRecognizer):
    """Nearby old keyframes (by estimated pose), ranked by how many local features they share."""

    name = "proximity"

    def __init__(self, config: RecognitionConfig, matcher: FeatureMatcher) -> None:
        super().__init__(config)
        self.matcher = matcher

    def _query(self, kf: Keyframe, old: list[Keyframe]) -> list[PlaceCandidate]:
        c = self.config
        dist = np.array([np.linalg.norm(k.pose.translation - kf.pose.translation) for k in old])
        near = [old[i] for i in np.argsort(dist)[: c.max_checked] if dist[i] < c.search_radius]
        out = []
        for cand in near:
            pairs = self.matcher.match(kf.features, cand.features)
            score = len(pairs) / max(len(kf.features), 1)
            if score >= c.min_vote_score:
                out.append(PlaceCandidate(cand, score, pairs))
        return sorted(out, key=lambda p: -p.score)


class GlobalRetrieval(PlaceRecognizer):
    """Nearest neighbours of the keyframe's global descriptor among *all* old keyframes (no pose prior)."""

    def __init__(self, config: RecognitionConfig, descriptor: GlobalDescriptor) -> None:
        super().__init__(config)
        self.descriptor = descriptor
        self.name = descriptor.name  # type: ignore[misc] - instance label = descriptor
        self.db: dict[int, FloatArray] = {}
        self.describe_ms: list[float] = []

    @property
    def device(self) -> str:
        return self.descriptor.device

    def add(self, kf: Keyframe) -> None:
        self.db[kf.id] = self.descriptor.describe(kf.frame.rgb)
        self.describe_ms.append(self.descriptor.last_ms)

    def _query(self, kf: Keyframe, old: list[Keyframe]) -> list[PlaceCandidate]:
        q = self.db.get(kf.id)
        if q is None:
            return []
        sims = np.array([float(self.db[k.id] @ q) for k in old])
        return [PlaceCandidate(old[i], float(sims[i])) for i in np.argsort(-sims)
                if sims[i] >= self.descriptor.min_similarity]


class PlaceRecognizerFactory:
    NAMES: ClassVar[tuple[str, ...]] = ("proximity", "thumbnail", "dinov2")

    @classmethod
    def create(cls, kind: str, config: RecognitionConfig, matcher: FeatureMatcher, device: str = "cpu"
               ) -> PlaceRecognizer:
        if kind == "proximity":
            return ProximityVoting(config, matcher)
        if kind in ("thumbnail", "dinov2"):
            return GlobalRetrieval(config, GlobalDescriptorFactory.create(kind, device))
        raise ValueError(f"unknown place_recognition '{kind}', expected one of {cls.NAMES}")
