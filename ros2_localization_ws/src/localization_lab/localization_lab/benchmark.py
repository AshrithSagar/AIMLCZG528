"""
Offline localization benchmark (no ROS) — the W2 "expected output" + the GPU evidence.

    ros2 run localization_lab train_pose_regressor --out ~/w2_models/pose_regressor.npz
    ros2 run localization_lab localization_benchmark --model ~/w2_models/pose_regressor.npz --out ~/w2_expected

Compares on one recorded run (global localization, then a kidnapping):
    grid     histogram filter over (x, y, theta)
    mcl      augmented Monte Carlo localization (Nav2 AMCL-style parameters)
    mcl_ml   MCL seeded / recovered with the learned scan->pose regressor
    ml_only  the regressor alone, one forward pass per scan
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .core.evaluation import LocalizationEvaluator, LocalizationLog  # noqa: E402
from .core.grid_localization import GridConfig, GridLocalizer  # noqa: E402
from .core.learned import ScanFeaturizer, ScanPoseRegressor  # noqa: E402
from .core.mcl import MclConfig, MonteCarloLocalizer  # noqa: E402
from .core.scenario import LabWorld, Recording, ScenarioConfig, ScenarioRecorder  # noqa: E402

DEFAULT_MODEL = Path.home() / "w2_models" / "pose_regressor.npz"


class Replayer:
    """Feeds a recording into a localizer and logs its error at every scan."""

    def __init__(self, rec: Recording) -> None:
        self.rec = rec

    def run_filter(self, loc, snapshots: tuple[float, ...] = ()) -> tuple[LocalizationLog, dict[float, Any]]:  # noqa: ANN001
        log, snaps = LocalizationLog(), {}
        if 0.0 in snapshots:
            snaps[0.0] = self._snapshot(loc)  # the prior: before any scan has been used
        prev = self.rec.samples[0].odom
        for s in self.rec.samples:
            est = loc.update(prev, s.odom, s.scan)
            prev = s.odom
            log.add(s.t, est.pose, s.truth, loc.stats.total_ms)
            for ts in snapshots:
                if ts not in snaps and s.t >= ts:
                    snaps[ts] = self._snapshot(loc)
        return log, snaps

    def run_regressor(self, reg: ScanPoseRegressor) -> LocalizationLog:
        log = LocalizationLog()
        for s in self.rec.samples:
            t0 = time.perf_counter()
            p = reg.propose(s.scan)
            log.add(s.t, p, s.truth, 1000 * (time.perf_counter() - t0))
        return log

    @staticmethod
    def _snapshot(loc) -> Any:  # noqa: ANN001, ANN401
        if isinstance(loc, GridLocalizer):
            return loc.belief_xy().copy()
        return loc.particles.copy()


class BenchmarkApp:
    SNAPSHOTS = (0.0, 0.2, 5.0)

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out = Path(args.out).expanduser()
        self.out.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        world = LabWorld.load(self.args.config_dir)
        rec = ScenarioRecorder(world, ScenarioConfig(duration_s=self.args.duration)).record()
        print(f"[scenario] {len(rec.samples)} scans, kidnapped at t={rec.kidnap_t:.1f}s to {rec.kidnap_to}")
        reg, train_report = self._regressor(world)
        rmax = world.robot.lidar.range_max
        replay, ev = Replayer(rec), LocalizationEvaluator()
        t0 = rec.samples[0].t
        snaps_at = self.SNAPSHOTS + ((rec.kidnap_t + 1.0),) if rec.kidnap_t else self.SNAPSHOTS
        results, logs, snaps = {}, {}, {}
        for name, factory in (
            ("grid", lambda: GridLocalizer(world.grid, rmax, GridConfig())),
            ("mcl", lambda: MonteCarloLocalizer(world.grid, rmax, MclConfig())),
            ("mcl_ml", lambda: self._mcl_ml(world, rmax, reg, rec)),
        ):
            print(f"[run] {name} ...", flush=True)
            logs[name], snaps[name] = replay.run_filter(factory(), snaps_at)
            results[name] = ev.report(logs[name], t0, rec.kidnap_t).as_dict()
            print(f"      {results[name]}")
        logs["ml_only"] = replay.run_regressor(reg)
        results["ml_only"] = ev.report(logs["ml_only"], t0, rec.kidnap_t).as_dict()
        results["ml_only"]["median_error_m"] = float(np.median(logs["ml_only"].pos_err))
        info = {"cpu_count": os.cpu_count(), "platform": platform.platform(), "python": platform.python_version()}
        payload = {"system": info, "scenario": {"scans": len(rec.samples), "kidnap_t": rec.kidnap_t},
                   "regressor_training": train_report, "results": results}
        (self.out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        Figures(self.out, world, rec, logs, snaps, snaps_at).write_all()
        (self.out / "summary.md").write_text(Summary.render(payload), encoding="utf-8")
        print((self.out / "summary.md").read_text())
        return payload

    def _regressor(self, world: LabWorld) -> tuple[ScanPoseRegressor, dict[str, float]]:
        path = Path(self.args.model).expanduser()
        if path.is_file():
            print(f"[model] loading {path}")
            return ScanPoseRegressor.load(path), {"loaded_from": str(path)}  # type: ignore[dict-item]
        print("[model] none found: training one now (CPU) ...")
        reg = ScanPoseRegressor(ScanFeaturizer(72, world.robot.lidar.range_max))
        rep = reg.fit_from_map(world.grid, world.robot.lidar, samples=self.args.samples, epochs=self.args.epochs)
        reg.save(path)
        return reg, rep.as_dict()

    @staticmethod
    def _mcl_ml(world: LabWorld, rmax: float, reg: ScanPoseRegressor, rec: Recording) -> MonteCarloLocalizer:
        mcl = MonteCarloLocalizer(world.grid, rmax, MclConfig(), proposal=reg)
        mcl.global_localize_with_proposal(rec.samples[0].scan)
        return mcl

    @staticmethod
    def main() -> None:
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("--out", default="w2_localization_benchmark")
        ap.add_argument("--model", default=str(DEFAULT_MODEL))
        ap.add_argument("--config-dir", default=None, help="W1 mobile_bot_description/config")
        ap.add_argument("--duration", type=float, default=120.0)
        ap.add_argument("--samples", type=int, default=40000)
        ap.add_argument("--epochs", type=int, default=60)
        BenchmarkApp(ap.parse_args()).run()


class TrainApp:
    """`train_pose_regressor`: learn scan -> pose from the map (CPU, ~1 min)."""

    @staticmethod
    def main() -> None:
        ap = argparse.ArgumentParser(description=TrainApp.__doc__)
        ap.add_argument("--out", default=str(DEFAULT_MODEL))
        ap.add_argument("--config-dir", default=None)
        ap.add_argument("--samples", type=int, default=40000)
        ap.add_argument("--epochs", type=int, default=60)
        a = ap.parse_args()
        world = LabWorld.load(a.config_dir)
        reg = ScanPoseRegressor(ScanFeaturizer(72, world.robot.lidar.range_max))
        print(f"Training on {a.samples} ray-cast scans ({os.cpu_count()} CPUs) ...")
        rep = reg.fit_from_map(world.grid, world.robot.lidar, samples=a.samples, epochs=a.epochs, verbose=True)
        print(json.dumps(rep.as_dict(), indent=2))
        print(f"saved {reg.save(a.out)}")


class Summary:
    @staticmethod
    def render(p: dict[str, Any]) -> str:
        fmt = lambda v, d=1: "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"  # noqa: E731
        r = p["results"]
        lines = [
            "# W2 localization benchmark",
            "",
            f"Machine: {p['system']['cpu_count']} CPUs ({p['system']['platform']}); "
            f"{p['scenario']['scans']} scans at 5 Hz, kidnapped at t = {fmt(p['scenario']['kidnap_t'])} s",
            "",
            "| method | converged in s | tracking RMSE cm | heading RMSE deg | recovered after kidnap in s | ms / update (mean, p95) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, m in r.items():
            lines.append(f"| {name} | {fmt(m['convergence_s'])} | {fmt(m['tracking_rmse_m'] * 100)} | "
                         f"{fmt(m['tracking_heading_rmse_deg'])} | {fmt(m['recovery_s']) if m['recovered'] else 'no'} | "
                         f"{fmt(m['update_ms_mean'], 2)}, {fmt(m['update_ms_p95'], 2)} |")
        tr = p.get("regressor_training", {})
        if "train_seconds" in tr:
            lines += ["", f"Regressor: {int(tr['params'])} parameters, trained on {int(tr['samples'])} scans in "
                          f"{tr['train_seconds']:.0f} s on CPU; validation median error {tr['val_median_pos_err_m'] * 100:.0f} cm "
                          f"(p90 {tr['val_p90_pos_err_m'] * 100:.0f} cm), inference {tr['inference_us']:.0f} µs/scan."]
        lines += ["", "Convergence = position error < 0.3 m and heading error < 15° held for 5 s."]
        return "\n".join(lines) + "\n"


class Figures:
    COLORS = {"grid": "tab:orange", "mcl": "tab:blue", "mcl_ml": "tab:green", "ml_only": "tab:red"}

    def __init__(self, out: Path, world: LabWorld, rec: Recording, logs: dict[str, LocalizationLog],
                 snaps: dict[str, dict[float, Any]], snap_times: tuple[float, ...]) -> None:
        self.out, self.world, self.rec, self.logs, self.snaps, self.snap_times = out, world, rec, logs, snaps, snap_times

    def write_all(self) -> None:
        self._errors()
        self._snapshots()

    def _errors(self) -> None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for name, log in self.logs.items():
            ax.semilogy(log.stamps, np.maximum(log.pos_err, 1e-3), lw=1.2 if name != "ml_only" else 0.6,
                        color=self.COLORS[name], label=name, alpha=1.0 if name != "ml_only" else 0.6)
        if self.rec.kidnap_t:
            ax.axvline(self.rec.kidnap_t, color="k", ls="--", label="kidnapped")
        ax.axhline(0.3, color="gray", ls=":", label="0.3 m")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("position error [m]")
        ax.legend(ncol=3)
        ax.set_title("Global localization, tracking, and recovery from kidnapping")
        fig.tight_layout()
        fig.savefig(self.out / "errors.png", dpi=130)
        plt.close(fig)

    def _snapshots(self) -> None:
        g = self.world.grid
        ext = (g.origin_x, g.origin_x + g.width * g.resolution, g.origin_y, g.origin_y + g.height * g.resolution)
        rows = [n for n in ("grid", "mcl", "mcl_ml") if n in self.snaps]
        fig, axes = plt.subplots(len(rows), len(self.snap_times), figsize=(4 * len(self.snap_times), 4 * len(rows)),
                                 squeeze=False)
        truth_at = {ts: next(s.truth for s in self.rec.samples if s.t >= max(ts, 1e-9)) for ts in self.snap_times}
        for r, name in enumerate(rows):
            for c, ts in enumerate(self.snap_times):
                ax = axes[r, c]
                ax.imshow(g.occupied, origin="lower", extent=ext, cmap="Greys", alpha=0.8)
                snap = self.snaps[name].get(ts)
                if snap is None:
                    continue
                if name == "grid":
                    ax.imshow(np.ma.masked_less(snap / snap.max(), 0.02), origin="lower", extent=ext, cmap="viridis",
                              alpha=0.85)
                else:
                    ax.scatter(snap[:, 0], snap[:, 1], s=3, c=self.COLORS[name], alpha=0.5, zorder=3)
                tp = truth_at[ts]
                ax.plot(tp.x, tp.y, "*", ms=9, mfc="none", mec="r", mew=1.5, zorder=4)
                label = "prior (before 1st scan)" if ts == 0.0 else f"t = {ts:.1f} s"
                ax.set_title(f"{name}  {label}" + ("  (after kidnap)" if self.rec.kidnap_t and ts > self.rec.kidnap_t else ""),
                             fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle("Belief snapshots (red star = true pose)")
        fig.tight_layout()
        fig.savefig(self.out / "beliefs.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    BenchmarkApp.main()
