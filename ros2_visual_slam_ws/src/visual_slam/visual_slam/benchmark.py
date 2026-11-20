"""
vslam_benchmark — runs the SLAM pipeline offline (no ROS) for several configurations
and answers: *what does each (learned) component buy, and is this machine fast enough?*

A configuration is `features[+matcher][@place_recognition]`, e.g.
    orb                         classical baseline
    superpoint+lightglue        learned features + learned matching
    orb@dinov2                  classical tracking, learned loop detection

    ros2 run visual_slam vslam_benchmark --out ~/vslam_bench                       # synthetic loop, no DL needed
    ros2 run visual_slam vslam_benchmark --source tum --path ~/datasets/rgbd_dataset_freiburg1_xyz
    ros2 run visual_slam vslam_benchmark --configs orb,superpoint+lightglue --device cuda
    ros2 run visual_slam vslam_benchmark --source tum --path ... --depth learned  # learned depth instead of sensor

    ros2 run visual_slam export_synthetic --out ~/datasets/synthetic_room   # TUM-format copy of the synthetic loop
    ros2 run visual_slam vslam_score --gt groundtruth.txt --est dpvo.txt --sim3   # score any TUM trajectory
    ros2 run visual_slam dl_check                                                 # do the learned models run here?

Outputs: summary.md, metrics.json, figures.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .core.camera import RgbdFrame  # noqa: E402
from .core.datasets import RgbdSequence, SequenceFactory, SyntheticSource, TumExporter  # noqa: E402
from .core.depth import DepthEstimatorFactory  # noqa: E402
from .core.metrics import (DepthEvaluator, PlaceRecognitionEvaluator, PlaceRecognitionReport, Trajectory,  # noqa: E402
                           TrajectoryEvaluator)
from .core.pipeline import SlamConfig, VisualSlamPipeline  # noqa: E402


class SystemInfo:
    """What the machine offers (the question the lab-machine check has to answer)."""

    @staticmethod
    def collect() -> dict[str, Any]:
        info: dict[str, Any] = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "opencv": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
        }
        try:
            import torch  # noqa: PLC0415

            info["torch"] = torch.__version__
            info["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                info["gpu"] = torch.cuda.get_device_name(0)
        except ImportError:
            info["torch"] = None
            info["cuda_available"] = False
        return info


@dataclass
class BackendRun:
    label: str
    device: str
    frames: int
    fps: float
    timings_mean_ms: dict[str, float]
    timings_p95_ms: dict[str, float]
    keyframes: int
    loops: int
    lost: int
    mean_features: float
    mean_inliers: float
    ate_vo_m: float | None
    ate_slam_m: float | None
    rpe_trans_m: float | None
    rpe_rot_deg: float | None
    place: PlaceRecognitionReport | None
    trajectory: Trajectory
    sample_keypoints: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("trajectory", "sample_keypoints", "place")}
        return d | {"place_recognition": self.place.as_dict() if self.place else None}


class BackendBenchmark:
    def __init__(self, seq: RgbdSequence, frames: list[RgbdFrame], gt: Trajectory,
                 depth_ms: list[float] | None = None) -> None:
        self.seq, self.frames, self.gt = seq, frames, gt
        self.depth_ms = depth_ms  # learned depth: per-frame inference cost, charged to every configuration
        self.evaluator = TrajectoryEvaluator()

    def run(self, cfg: SlamConfig) -> BackendRun:
        pipe = VisualSlamPipeline(self.seq.camera, cfg)
        live = Trajectory.empty()
        stages: dict[str, list[float]] = {}
        feats, inl, loops, lost = [], [], 0, 0
        t0 = time.perf_counter()
        for i, f in enumerate(self.frames):
            r = pipe.process(f)
            if self.depth_ms is not None:
                r.timings_ms["depth_ms"] = self.depth_ms[i]
            live.add(f.stamp, r.pose)
            for k, v in r.timings_ms.items():
                stages.setdefault(k, []).append(v)
            feats.append(r.features)
            inl.append(r.inliers)
            loops += r.loop_closed
            lost += r.state.value == "lost"
        wall = time.perf_counter() - t0 + (sum(self.depth_ms) / 1000.0 if self.depth_ms is not None else 0.0)
        slam = pipe.trajectory()
        vo_rep = self.evaluator.evaluate(self.gt, live) if len(self.gt) else None
        slam_rep = self.evaluator.evaluate(self.gt, slam) if len(self.gt) else None
        n = len(self.frames)
        place = PlaceRecognitionEvaluator(min_separation=cfg.loop.recognition.min_separation).evaluate(
            pipe.loops.queries, [k.stamp for k in pipe.map.keyframes], self.gt) if len(self.gt) else None
        return BackendRun(
            cfg.label, pipe.extractor.device, n, n / wall,
            {k: float(np.sum(v) / n) for k, v in stages.items()},  # amortised per frame
            {k: float(np.percentile(v, 95)) for k, v in stages.items()},
            len(pipe.map.keyframes), loops, lost, float(np.mean(feats)), float(np.mean(inl)),
            vo_rep.ate_rmse_m if vo_rep else None, slam_rep.ate_rmse_m if slam_rep else None,
            slam_rep.rpe_trans_rmse_m if slam_rep else None, slam_rep.rpe_rot_rmse_deg if slam_rep else None,
            place, slam, pipe.extractor.extract(self.frames[len(self.frames) // 2].gray).keypoints,
        )


class ReportWriter:
    STAGES = ("depth_ms", "extract_ms", "match_ms", "pnp_ms", "place_db_ms", "loop_ms", "pgo_ms")

    def __init__(self, out: Path, seq: RgbdSequence, frames: list[RgbdFrame], gt: Trajectory,
                 input_hz: float, info: dict[str, Any], depth: dict[str, Any] | None = None) -> None:
        self.out, self.seq, self.frames, self.gt, self.input_hz, self.info = out, seq, frames, gt, input_hz, info
        self.depth = depth  # learned-depth summary (None = sensor depth)
        out.mkdir(parents=True, exist_ok=True)

    def write(self, runs: list[BackendRun], skipped: dict[str, str]) -> None:
        self._trajectories(runs)
        self._timings(runs)
        self._keypoints(runs)
        (self.out / "metrics.json").write_text(json.dumps(
            {"system": self.info, "sequence": self.seq.name, "input_hz": self.input_hz,
             "depth": self.depth, "runs": [r.as_dict() for r in runs], "skipped": skipped}, indent=2),
            encoding="utf-8")
        (self.out / "summary.md").write_text(self._summary(runs, skipped), encoding="utf-8")
        for r in runs:
            r.trajectory.write_tum(self.out / f"estimate_{r.label}.tum")
        if len(self.gt):
            self.gt.write_tum(self.out / "ground_truth.tum")

    def _summary(self, runs: list[BackendRun], skipped: dict[str, str]) -> str:
        fmt = lambda v, s=100: "—" if v is None else f"{v * s:.1f}"  # noqa: E731
        lines = [
            f"# VSLAM benchmark — {self.seq.name}",
            "",
            f"Machine: {self.info['cpu_count']} CPUs, OpenCV {self.info['opencv']}, "
            f"torch {self.info['torch']}, CUDA {'yes (' + self.info.get('gpu', '') + ')' if self.info['cuda_available'] else 'no'}",
            f"Input: {len(self.frames)} frames, {self.seq.camera.width}x{self.seq.camera.height} @ {self.input_hz:.0f} Hz, "
            f"depth: {'learned (' + self.depth['model'] + '; fps includes its inference time)' if self.depth else 'sensor'}",
            "",
            "Configuration = `features[+matcher][@place_recognition]` (defaults: brute-force matching, proximity loop search).",
            "",
            "| configuration | device | fps | real-time? | extract ms | match ms | PnP ms | loop+PGO ms | ATE VO cm | ATE SLAM cm | RPE cm/s | KFs | loops | lost |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in runs:
            t = r.timings_mean_ms
            rt = "yes" if r.fps >= self.input_hz else f"no ({r.fps / self.input_hz:.0%})"
            lines.append(
                f"| {r.label} | {r.device} | {r.fps:.1f} | {rt} | {t.get('extract_ms', 0):.1f} | "
                f"{t.get('match_ms', 0):.1f} | {t.get('pnp_ms', 0):.1f} | "
                f"{t.get('place_db_ms', 0) + t.get('loop_ms', 0) + t.get('pgo_ms', 0):.1f} | "
                f"{fmt(r.ate_vo_m)} | {fmt(r.ate_slam_m)} | {fmt(r.rpe_trans_m)} | {r.keyframes} | {r.loops} | {r.lost} |")
        for k, why in skipped.items():
            lines.append(f"| {k} | — | skipped: {why} | | | | | | | | | | | |")
        lines += ["", "Real-time = the pipeline keeps up with the camera rate on this machine.",
                  "If a learned component is not real-time on CPU but is with `--device cuda`, that is the case for a GPU."]
        lines += self._place_section(runs) + self._depth_section()
        return "\n".join(lines) + "\n"

    def _place_section(self, runs: list[BackendRun]) -> list[str]:
        rows = [r for r in runs if r.place is not None]
        if not rows:
            return []
        out = ["", "## Loop closure: place recognition", "",
               "Ground truth: same place = true poses within 1 m and 30°. Recall@1 counts queries whose best "
               "candidate (before the geometric check) is a true revisit; false loops = accepted but wrong.", "",
               "| configuration | queries | with a true revisit | recall@1 | loops accepted | false loops | ms / query |",
               "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            p = r.place
            assert p is not None
            rec = "—" if np.isnan(p.recall_at_1) else f"{p.recall_at_1:.2f}"
            out.append(f"| {r.label} | {p.queries} | {p.queries_with_revisit} | {rec} | {p.loops_accepted} | "
                       f"{p.false_loops} | {p.ms_per_query:.1f} |")
        return out

    def _depth_section(self) -> list[str]:
        if not self.depth:
            return []
        d = self.depth
        return ["", "## Learned depth vs sensor depth", "",
                "NYU-Depth-v2 / KITTI protocol (Eigen et al. 2014). median scale = 1.0 means the metric scale is right.",
                "", "| model | device | ms / frame | AbsRel | RMSE m | δ<1.25 | median scale |",
                "| --- | --- | --- | --- | --- | --- | --- |",
                f"| {d['model']} | {d['device']} | {d['ms_mean']:.0f} | {d['abs_rel']:.3f} | {d['rmse_m']:.3f} | "
                f"{d['delta_1']:.3f} | {d['median_scale']:.2f} |"]

    def _trajectories(self, runs: list[BackendRun]) -> None:
        fig, ax = plt.subplots(figsize=(7, 7))
        ev = TrajectoryEvaluator()
        if len(self.gt):
            g = self.gt.positions
            ax.plot(g[:, 0], g[:, 1], "k-", lw=2, label="ground truth")
            for r in runs:
                _, aligned = ev.aligned_positions(self.gt, r.trajectory)
                ax.plot(aligned[:, 0], aligned[:, 1], lw=1.2,
                        label=f"{r.label} (ATE {r.ate_slam_m * 100:.1f} cm)" if r.ate_slam_m else r.label)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend()
        ax.set_title(f"Trajectories (SE(3)-aligned) — {self.seq.name}")
        fig.tight_layout()
        fig.savefig(self.out / "trajectories.png", dpi=130)
        plt.close(fig)

    def _timings(self, runs: list[BackendRun]) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bottom = np.zeros(len(runs))
        names = [f"{r.label}\n({r.device})" for r in runs]
        for st in self.STAGES:
            vals = np.array([r.timings_mean_ms.get(st, 0.0) for r in runs])
            ax.bar(names, vals, bottom=bottom, label=st.replace("_ms", ""))
            bottom += vals
        ax.axhline(1000.0 / self.input_hz, c="r", ls="--", label=f"camera budget ({self.input_hz:.0f} Hz)")
        ax.set_ylabel("ms per frame (mean)")
        ax.legend()
        ax.set_title("Where the time goes")
        fig.tight_layout()
        fig.savefig(self.out / "timings.png", dpi=130)
        plt.close(fig)

    def _keypoints(self, runs: list[BackendRun]) -> None:
        f = self.frames[len(self.frames) // 2]
        fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4), squeeze=False)
        for ax, r in zip(axes[0], runs):
            ax.imshow(f.rgb)
            ax.scatter(r.sample_keypoints[:, 0], r.sample_keypoints[:, 1], s=4, c="lime")
            ax.set_title(f"{r.label}: {len(r.sample_keypoints)} keypoints")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(self.out / "keypoints.png", dpi=110)
        plt.close(fig)


class LearnedDepthStage:
    """Replaces every frame's sensor depth with the network's prediction and scores it against the sensor."""

    def __init__(self, model_id: str, device: str) -> None:
        self.estimator = DepthEstimatorFactory.create("depth_anything", model_id, device)
        self.evaluator = DepthEvaluator()

    def apply(self, frames: list[RgbdFrame]) -> tuple[list[RgbdFrame], list[float], dict[str, Any]]:
        out, ms, reports = [], [], []
        for i, f in enumerate(frames):
            pred = self.estimator.frame(f)
            ms.append(self.estimator.last_ms)
            if i % 5 == 0:
                reports.append(self.evaluator.evaluate(pred.depth, f.depth))
            out.append(pred)
            if i % 50 == 0:
                print(f"      depth {i}/{len(frames)}: {self.estimator.last_ms:.0f} ms/frame", flush=True)
        summary = {k: float(np.nanmean([r.as_dict()[k] for r in reports]))
                   for k in ("abs_rel", "rmse_m", "delta_1", "median_scale")}
        return out, ms, summary | {"model": self.estimator.name, "device": self.estimator.device,
                                   "ms_mean": float(np.mean(ms))}


class BenchmarkApp:
    DEFAULT_CONFIGS = "orb,sift,akaze,superpoint,superpoint+lightglue,orb@thumbnail,orb@dinov2"

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def run(self) -> None:
        a = self.args
        info = SystemInfo.collect()
        print(f"[system] {info}")
        seq = SequenceFactory.create(a.source, a.path)
        n = len(seq) if a.max_frames <= 0 else min(len(seq), a.max_frames)
        print(f"[data] {seq.name}: loading {n} frames ...")
        frames = [seq.frame(i) for i in range(n)]
        gt = Trajectory.empty()
        for i, f in enumerate(frames):
            p = seq.ground_truth(i)
            if p is not None:
                gt.add(f.stamp, p)
        input_hz = (n - 1) / (frames[-1].stamp - frames[0].stamp) if n > 1 else 0.0
        depth_ms, depth = None, None
        if a.depth == "learned":
            print(f"[depth] predicting depth for {n} frames ...", flush=True)
            frames, depth_ms, depth = LearnedDepthStage(a.depth_model_id, a.device).apply(frames)
        bench = BackendBenchmark(seq, frames, gt, depth_ms)
        runs, skipped = [], {}
        for label in [s.strip() for s in a.configs.split(",") if s.strip()]:
            try:
                print(f"[run] {label} ...", flush=True)
                r = bench.run(SlamConfig.from_label(label, device=a.device))
                runs.append(r)
                print(f"      {r.fps:.1f} fps, ATE SLAM {r.ate_slam_m}, loops {r.loops}")
            except (RuntimeError, ValueError) as exc:
                skipped[label] = str(exc)
                print(f"      skipped: {exc}")
        out = Path(a.out).expanduser()
        ReportWriter(out, seq, frames, gt, input_hz, info, depth).write(runs, skipped)
        print((out / "summary.md").read_text())

    @staticmethod
    def main() -> None:
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("--source", choices=["synthetic", "tum"], default="synthetic")
        ap.add_argument("--path", default="", help="TUM sequence folder (for --source tum)")
        ap.add_argument("--configs", default=BenchmarkApp.DEFAULT_CONFIGS,
                        help="comma list of features[+matcher][@place_recognition]")
        ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
        ap.add_argument("--depth", default="sensor", choices=["sensor", "learned"])
        ap.add_argument("--depth-model-id", default="", help="Hugging Face model id for --depth learned")
        ap.add_argument("--max-frames", type=int, default=0)
        ap.add_argument("--out", default="vslam_benchmark")
        BenchmarkApp(ap.parse_args()).run()


class ExportApp:
    @staticmethod
    def main() -> None:
        ap = argparse.ArgumentParser(description="Write the synthetic RGB-D loop in TUM RGB-D format.")
        ap.add_argument("--out", default=str(Path.home() / "datasets" / "synthetic_room"))
        a = ap.parse_args()
        seq = SyntheticSource()
        print(f"Rendering {len(seq)} frames ({seq.camera.width}x{seq.camera.height}) to {a.out} ...")
        print(f"Done: {TumExporter.write(seq, a.out)}")


if __name__ == "__main__":
    BenchmarkApp.main()
