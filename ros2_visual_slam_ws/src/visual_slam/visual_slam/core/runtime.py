"""
Optional deep-learning runtime.

Every learned component (SuperPoint, LightGlue, DINOv2, Depth Anything) goes
through `TorchRuntime`, so:

  * PyTorch stays an *optional* dependency — the classical pipeline never imports it,
  * a missing package fails with one clear "pip install ..." message,
  * `device:=cuda` silently falls back to CPU when there is no GPU (and says so),
  * timing includes GPU synchronisation, so CPU-vs-GPU numbers are honest.
"""

from __future__ import annotations

import importlib
import time
from types import ModuleType
from typing import Any


class MissingDependency(RuntimeError):
    """A learned component was requested but its Python package is not installed."""


class TorchRuntime:
    INSTALL_HINTS: dict[str, str] = {
        "torch": "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   (CPU build)",
        "lightglue": "pip install git+https://github.com/cvg/LightGlue.git",
        "transformers": "pip install 'transformers>=4.45'",
    }

    def __init__(self, device: str = "cpu") -> None:
        self.torch = self.require("torch")
        self.requested = device
        self.device = device if device == "cpu" or self.torch.cuda.is_available() else "cpu"

    @property
    def fell_back(self) -> bool:
        return self.device != self.requested

    @classmethod
    def require(cls, module: str) -> ModuleType:
        try:
            return importlib.import_module(module)
        except ImportError as exc:
            hint = cls.INSTALL_HINTS.get(module.split(".")[0], f"pip install {module}")
            raise MissingDependency(f"'{module}' is not installed: {hint}") from exc

    @classmethod
    def available(cls, module: str) -> bool:
        try:
            cls.require(module)
        except MissingDependency:
            return False
        return True

    def tensor(self, array: Any) -> Any:  # noqa: ANN401 - numpy in, torch tensor out
        return self.torch.from_numpy(array).to(self.device)

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def no_grad(self) -> Any:  # noqa: ANN401 - torch context manager
        return self.torch.inference_mode()


class Stopwatch:
    """Context manager measuring wall time in ms (GPU-synchronised when a runtime is given)."""

    def __init__(self, runtime: TorchRuntime | None = None) -> None:
        self.runtime = runtime
        self.ms = 0.0
        self._t0 = 0.0

    def __enter__(self) -> Stopwatch:
        if self.runtime:
            self.runtime.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.runtime:
            self.runtime.synchronize()
        self.ms = 1000.0 * (time.perf_counter() - self._t0)
