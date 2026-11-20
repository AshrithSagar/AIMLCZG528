"""
Localization metrics (shared by the offline benchmark and the ROS evaluator node).

  convergence time   first time the error stays below (0.3 m, 15 deg) for `hold` seconds
  tracking RMSE      position RMSE once converged
  recovery time      the same criterion, measured from a kidnapping event
  update cost        ms per filter update (the "do we need a GPU?" number)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np

from sensor_motion_lab.core import Angle, Pose2D


@dataclass(frozen=True, slots=True)
class ConvergenceCriterion:
    position_m: float = 0.3
    heading_deg: float = 15.0
    hold_s: float = 5.0


@dataclass
class LocalizationLog:
    stamps: list[float] = field(default_factory=list)
    pos_err: list[float] = field(default_factory=list)
    head_err_deg: list[float] = field(default_factory=list)
    update_ms: list[float] = field(default_factory=list)
    estimates: list[Pose2D] = field(default_factory=list)

    def add(self, t: float, est: Pose2D, truth: Pose2D, ms: float = 0.0) -> None:
        self.stamps.append(t)
        self.pos_err.append(math.hypot(est.x - truth.x, est.y - truth.y))
        self.head_err_deg.append(math.degrees(abs(Angle.wrap(est.theta - truth.theta))))
        self.update_ms.append(ms)
        self.estimates.append(est)


@dataclass(frozen=True, slots=True)
class LocalizationReport:
    converged: bool
    convergence_s: float
    tracking_rmse_m: float
    tracking_heading_rmse_deg: float
    recovered: bool
    recovery_s: float
    update_ms_mean: float
    update_ms_p95: float

    def as_dict(self) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}


class LocalizationEvaluator:
    def __init__(self, criterion: ConvergenceCriterion = ConvergenceCriterion()) -> None:
        self.c = criterion

    def first_converged(self, log: LocalizationLog, t_from: float, t_to: float = math.inf) -> float | None:
        t = np.asarray(log.stamps)
        ok = (np.asarray(log.pos_err) < self.c.position_m) & (np.asarray(log.head_err_deg) < self.c.heading_deg)
        win = (t >= t_from) & (t < t_to)
        idx = np.nonzero(win)[0]
        for k, i in enumerate(idx):
            j = idx[k:]
            hold = j[t[j] <= t[i] + self.c.hold_s]
            if t[hold[-1]] - t[i] >= self.c.hold_s - 1e-9 and ok[hold].all():
                return float(t[i])
        return None

    def report(self, log: LocalizationLog, t_start: float, kidnap_t: float | None = None) -> LocalizationReport:
        t = np.asarray(log.stamps)
        end_first = kidnap_t if kidnap_t is not None else math.inf
        tc = self.first_converged(log, t_start, end_first)
        track = (t >= tc) & (t < end_first) if tc is not None else np.zeros_like(t, dtype=bool)
        pe, he = np.asarray(log.pos_err), np.asarray(log.head_err_deg)
        tr = self.first_converged(log, kidnap_t) if kidnap_t is not None else None
        ms = np.asarray(log.update_ms)
        return LocalizationReport(
            converged=tc is not None,
            convergence_s=(tc - t_start) if tc is not None else float("nan"),
            tracking_rmse_m=float(np.sqrt(np.mean(pe[track] ** 2))) if track.any() else float("nan"),
            tracking_heading_rmse_deg=float(np.sqrt(np.mean(he[track] ** 2))) if track.any() else float("nan"),
            recovered=tr is not None,
            recovery_s=(tr - kidnap_t) if (tr is not None and kidnap_t is not None) else float("nan"),
            update_ms_mean=float(ms.mean()) if ms.size else float("nan"),
            update_ms_p95=float(np.percentile(ms, 95)) if ms.size else float("nan"),
        )
