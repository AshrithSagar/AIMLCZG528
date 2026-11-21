"""
Learned monocular depth: colour image -> metric depth image.

With it, a plain RGB camera (a webcam, a phone) can feed the *RGB-D* SLAM
pipeline unchanged: the depth node publishes the same `/camera/depth/image_rect_raw`
topic an RGB-D driver would. What changes is the quality of that depth — measured
with the standard AbsRel / RMSE / δ<1.25 metrics against the dataset's sensor depth.

  depth_anything  Depth Anything V2, metric indoor fine-tune (Yang et al., NeurIPS 2024), via `transformers`
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .camera import RgbdFrame
from .runtime import TorchRuntime


class DepthEstimator(ABC):
    name: ClassVar[str] = "base"

    def __init__(self) -> None:
        self.last_ms = 0.0

    def estimate(self, rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
        """(H, W, 3) RGB -> (H, W) float32 metres, same resolution as the input."""
        t0 = time.perf_counter()
        depth = np.asarray(self._estimate(rgb), dtype=np.float32)
        self.last_ms = 1000.0 * (time.perf_counter() - t0)
        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"{self.name}: depth {depth.shape} does not match the image {rgb.shape[:2]}")
        return depth

    def frame(self, frame: RgbdFrame) -> RgbdFrame:
        """The same frame with its sensor depth replaced by the prediction."""
        return RgbdFrame(frame.stamp, frame.rgb, self.estimate(frame.rgb))

    @abstractmethod
    def _estimate(self, rgb: NDArray[np.uint8]) -> NDArray[np.float32]: ...

    @property
    def device(self) -> str:
        return "cpu"


class DepthAnythingEstimator(DepthEstimator):
    name = "depth_anything"
    DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        super().__init__()
        self.rt = TorchRuntime(device)
        transformers = TorchRuntime.require("transformers")
        self.processor = transformers.AutoImageProcessor.from_pretrained(model_id)
        self.net = transformers.AutoModelForDepthEstimation.from_pretrained(model_id).eval().to(self.rt.device)
        self.model_id = model_id

    @property
    def device(self) -> str:
        return self.rt.device

    def _estimate(self, rgb: NDArray[np.uint8]) -> NDArray[np.float32]:
        torch = self.rt.torch
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.rt.device)
        with self.rt.no_grad():
            pred = self.net(**inputs).predicted_depth  # (1, h, w) at the network's resolution
            pred = torch.nn.functional.interpolate(pred[:, None], size=rgb.shape[:2], mode="bicubic",
                                                   align_corners=False)[0, 0]
        return pred.clamp(min=0.0).cpu().numpy()


class DepthEstimatorFactory:
    NAMES: ClassVar[tuple[str, ...]] = ("depth_anything",)

    @classmethod
    def create(cls, kind: str = "depth_anything", model_id: str = "", device: str = "cpu") -> DepthEstimator:
        if kind == "depth_anything":
            return DepthAnythingEstimator(model_id or DepthAnythingEstimator.DEFAULT_MODEL, device)
        raise ValueError(f"unknown depth estimator '{kind}', expected one of {cls.NAMES}")
