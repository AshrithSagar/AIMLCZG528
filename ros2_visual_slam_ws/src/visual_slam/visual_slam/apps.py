"""
Command-line tools around the learned components.

    ros2 run visual_slam dl_check [--device cuda]        do the four learned models run here, and how fast?
    ros2 run visual_slam vslam_score --gt <TUM folder or groundtruth.txt> --est a.tum [b.tum ...] [--sim3]

`dl_check` needs no dataset (it uses the synthetic loop) and downloads each model
once (Hugging Face / GitHub). Paste its output back to whoever is preparing the lab.
"""

from __future__ import annotations

import argparse
import platform
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .core.camera import RgbdFrame
from .core.datasets import SyntheticSource
from .core.depth import DepthEstimatorFactory
from .core.features import FeatureExtractorFactory, SuperPointExtractor
from .core.matching import LightGlueMatcher
from .core.metrics import DepthEvaluator, FrameIndexStamps, Trajectory, TrajectoryEvaluator
from .core.place_recognition import Dinov2Descriptor
from .core.runtime import MissingDependency, TorchRuntime
from .core.tracking import PnpSolver


# --------------------------------------------------------------------------- #
# dl_check                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class CheckResult:
    name: str
    status: str  # PASS | FAIL | SKIP
    metrics: dict[str, Any] = field(default_factory=dict)
    note: str = ""


class CheckContext:
    """Synthetic frames shared by every check: two nearby views, a revisit, and the opposite side of the loop."""

    REPEATS = 5

    def __init__(self, device: str) -> None:
        self.device = device
        self.seq = SyntheticSource()
        per_loop = int(round(self.seq.seq.trajectory.period * self.seq.seq.trajectory.rate_hz))
        self.idx = {"a": 0, "b": 8, "revisit": per_loop, "opposite": per_loop // 2}
        self.frames: dict[str, RgbdFrame] = {k: self.seq.frame(i) for k, i in self.idx.items()}

    def timed(self, fn: Any, *args: Any) -> tuple[Any, float]:  # noqa: ANN401
        fn(*args)  # warm-up (first call allocates / compiles)
        t0 = time.perf_counter()
        for _ in range(self.REPEATS):
            out = fn(*args)
        return out, 1000.0 * (time.perf_counter() - t0) / self.REPEATS


class DlCheck(ABC):
    name: ClassVar[str] = "base"

    def run(self, ctx: CheckContext) -> CheckResult:
        try:
            return self._run(ctx)
        except MissingDependency as exc:
            return CheckResult(self.name, "SKIP", note=str(exc))
        except Exception as exc:  # noqa: BLE001 - report every failure, keep checking the rest
            return CheckResult(self.name, "FAIL", note=f"{type(exc).__name__}: {exc}")

    @abstractmethod
    def _run(self, ctx: CheckContext) -> CheckResult: ...


class SuperPointLightGlueCheck(DlCheck):
    """Learned features + matching: relative pose between two views vs ground truth."""

    name = "superpoint + lightglue"

    def _run(self, ctx: CheckContext) -> CheckResult:
        sp = SuperPointExtractor(1000, ctx.device)
        lg = LightGlueMatcher("superpoint", ctx.device)
        fa, fb = ctx.frames["a"], ctx.frames["b"]
        ka, sp_ms = ctx.timed(sp.extract, fa.gray)
        kb = sp.extract(fb.gray)
        pairs, lg_ms = ctx.timed(lg.match, kb, ka)
        cam = ctx.seq.camera
        d = fa.depth_at(ka.keypoints[pairs[:, 1]])
        ok = d > 0.2
        res = PnpSolver(cam).solve(cam.backproject(ka.keypoints[pairs[ok, 1]], d[ok]), kb.keypoints[pairs[ok, 0]])
        true = ctx.seq.ground_truth(ctx.idx["b"]).inverse() @ ctx.seq.ground_truth(ctx.idx["a"])
        err_cm = 100 * float(np.linalg.norm(res.T_cur_ref.translation - true.translation)) if res.ok else float("nan")
        status = "PASS" if res.ok and len(pairs) >= 50 and err_cm < 5 else "FAIL"
        return CheckResult(self.name, status, {"device": sp.device, "keypoints": len(ka), "matches": len(pairs),
                                               "pnp_inliers": len(res.inliers), "pose_error_cm": round(err_cm, 2),
                                               "superpoint_ms": round(sp_ms, 1), "lightglue_ms": round(lg_ms, 1)})


class Dinov2Check(DlCheck):
    """Learned place recognition: a revisit must look more similar than the far side of the room."""

    name = "dinov2 place recognition"

    def _run(self, ctx: CheckContext) -> CheckResult:
        desc = Dinov2Descriptor(device=ctx.device)
        a, ms = ctx.timed(desc.describe, ctx.frames["a"].rgb)
        sim_rev = float(a @ desc.describe(ctx.frames["revisit"].rgb))
        sim_opp = float(a @ desc.describe(ctx.frames["opposite"].rgb))
        status = "PASS" if sim_rev > sim_opp else "FAIL"
        return CheckResult(self.name, status, {"device": desc.device, "dim": int(a.size), "sim_revisit": round(sim_rev, 3),
                                               "sim_opposite": round(sim_opp, 3), "ms_per_keyframe": round(ms, 1)})


class DepthAnythingCheck(DlCheck):
    """Learned depth vs the synthetic sensor depth (a sanity check; real numbers come from TUM)."""

    name = "depth anything v2 (metric indoor)"

    def _run(self, ctx: CheckContext) -> CheckResult:
        est = DepthEstimatorFactory.create("depth_anything", device=ctx.device)
        f = ctx.frames["a"]
        pred, ms = ctx.timed(est.estimate, f.rgb)
        rep = DepthEvaluator().evaluate(pred, f.depth)
        status = "PASS" if np.isfinite(rep.abs_rel) and pred.shape == f.depth.shape else "FAIL"
        return CheckResult(self.name, status, {"device": est.device, "ms_per_frame": round(ms, 1),
                                               "abs_rel": round(rep.abs_rel, 3), "delta_1": round(rep.delta_1, 3),
                                               "median_scale": round(rep.median_scale, 2)},
                           note="synthetic textures are unlike real rooms: judge accuracy on TUM, not here")


class ClassicalBaselineCheck(DlCheck):
    """ORB on the same pair, so the learned timings have a reference on this machine."""

    name = "orb (classical baseline)"

    def _run(self, ctx: CheckContext) -> CheckResult:
        orb = FeatureExtractorFactory.create("orb")
        _, ms = ctx.timed(orb.extract, ctx.frames["a"].gray)
        return CheckResult(self.name, "PASS", {"extract_ms": round(ms, 1)})


class DlCheckApp:
    CHECKS: ClassVar[tuple[type[DlCheck], ...]] = (ClassicalBaselineCheck, SuperPointLightGlueCheck, Dinov2Check,
                                                   DepthAnythingCheck)

    def __init__(self, device: str, out: Path) -> None:
        self.device, self.out = device, out

    def environment(self) -> dict[str, Any]:
        env: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
        for mod in ("torch", "lightglue", "transformers"):
            try:
                m = TorchRuntime.require(mod)
                env[mod] = getattr(m, "__version__", "installed")
            except MissingDependency:
                env[mod] = None
        if env["torch"]:
            torch = TorchRuntime.require("torch")
            env["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            env["torch_threads"] = torch.get_num_threads()
        return env

    def run(self) -> None:
        env = self.environment()
        print(f"[env] {env}")
        ctx = CheckContext(self.device)
        results = []
        for check in self.CHECKS:
            print(f"[check] {check.name} ...", flush=True)
            r = check().run(ctx)
            print(f"        {r.status} {r.metrics} {r.note}")
            results.append(r)
        report = self.markdown(env, results)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(report, encoding="utf-8")
        print("\n" + report + f"\n(written to {self.out})")

    def markdown(self, env: dict[str, Any], results: list[CheckResult]) -> str:
        lines = ["# dl_check", "", f"Requested device: {self.device}", "",
                 *(f"- {k}: {v}" for k, v in env.items()), "",
                 "| check | status | metrics | note |", "| --- | --- | --- | --- |"]
        lines += [f"| {r.name} | {r.status} | {', '.join(f'{k}={v}' for k, v in r.metrics.items())} | {r.note} |"
                  for r in results]
        return "\n".join(lines) + "\n"

    @staticmethod
    def main() -> None:
        ap = argparse.ArgumentParser(description="Check that the learned W3 components run on this machine.")
        ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
        ap.add_argument("--out", default=str(Path.home() / "vslam_results" / "dl_check.md"))
        a = ap.parse_args()
        DlCheckApp(a.device, Path(a.out).expanduser()).run()


# --------------------------------------------------------------------------- #
# vslam_score                                                                 #
# --------------------------------------------------------------------------- #
class ScoreApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def ground_truth(self) -> Trajectory:
        p = Path(self.args.gt).expanduser()
        return Trajectory.read_tum(p / "groundtruth.txt" if p.is_dir() else p)

    def run(self) -> None:
        a = self.args
        gt = self.ground_truth()
        evaluator = TrajectoryEvaluator(max_dt=a.max_dt, with_scale=a.sim3)
        remap = FrameIndexStamps(Path(a.frame_stamps).expanduser(), a.stride) if a.frame_stamps else None
        print(f"Ground truth: {len(gt)} poses; alignment {'Sim(3) (monocular)' if a.sim3 else 'SE(3)'}\n")
        print("| trajectory | poses | matched | ATE RMSE cm | ATE max cm | RPE trans cm | RPE rot deg |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        for path in a.est:
            est = Trajectory.read_tum(path)
            if FrameIndexStamps.looks_like_indices(est):
                if remap is None:
                    print(f"| {Path(path).name} | {len(est)} | 0 | stamped with frame indices: add "
                          "--frame-stamps <seq>/rgb.txt --stride <the run's stride> | | | |")
                    continue
                est = remap.apply(est)
            rep = evaluator.evaluate(gt, est)
            if rep is None:
                print(f"| {Path(path).name} | {len(est)} | 0 | timestamps do not match the ground truth | | | |")
                continue
            print(f"| {Path(path).name} | {len(est)} | {rep.pairs} | {100 * rep.ate_rmse_m:.1f} | "
                  f"{100 * rep.ate_max_m:.1f} | {100 * rep.rpe_trans_rmse_m:.1f} | {rep.rpe_rot_rmse_deg:.2f} |")

    @staticmethod
    def main() -> None:
        ap = argparse.ArgumentParser(description="TUM ATE/RPE of any trajectory file(s) against ground truth.")
        ap.add_argument("--gt", required=True, help="groundtruth.txt or a TUM sequence folder")
        ap.add_argument("--est", required=True, nargs="+", help="TUM-format trajectories")
        ap.add_argument("--sim3", action="store_true", help="align with scale (monocular methods)")
        ap.add_argument("--max-dt", type=float, default=0.02)
        ap.add_argument("--frame-stamps", default="", help="rgb.txt, if the estimate is stamped with frame indices")
        ap.add_argument("--stride", type=int, default=1, help="frame stride used by the DL SLAM run")
        ScoreApp(ap.parse_args()).run()
