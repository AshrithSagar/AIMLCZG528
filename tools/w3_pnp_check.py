"""
W3 tracking check: the repo's PnP solver vs the previous one (RANSAC seeded with the motion prior),
on the same frames, on this machine's OpenCV.

    source /opt/ros/jazzy/setup.bash && source ros2_visual_slam_ws/install/setup.bash
    python3 tools/w3_pnp_check.py [--tum ~/datasets/rgbd_dataset_freiburg1_xyz]

Prints the ATE every 100 frames and, for any solver that drifts, the first frames where the estimated
frame-to-frame motion disagrees with the ground truth by more than 5 cm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from visual_slam.core.datasets import RgbdSequence, SyntheticSource, TumSequence
from visual_slam.core.metrics import Trajectory, TrajectoryEvaluator
from visual_slam.core.pipeline import SlamConfig, VisualSlamPipeline
from visual_slam.core.se3 import SE3
from visual_slam.core.tracking import PnpResult, PnpSolver


class PreviousPnpSolver(PnpSolver):
    """The solver as it was: OpenCV's iterative PnP inside RANSAC, seeded with the motion prior."""

    def solve(self, pts3d, pts2d, guess=None):  # noqa: ANN001, ANN201
        if pts3d.shape[0] < max(self.min_inliers, 6):
            return PnpResult(SE3.identity(), np.zeros(0, dtype=np.intp), False)
        rvec0 = guess.rvec().reshape(3, 1).copy() if guess else np.zeros((3, 1))
        tvec0 = guess.translation.reshape(3, 1).copy() if guess else np.zeros((3, 1))
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            pts3d.astype(np.float64), pts2d.astype(np.float64), self.camera.k, None, rvec=rvec0, tvec=tvec0,
            useExtrinsicGuess=guess is not None, iterationsCount=self.iterations,
            reprojectionError=self.reproj_error_px, confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inl is None or len(inl) < self.min_inliers:
            return PnpResult(SE3.identity(), np.zeros(0, dtype=np.intp), False)
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(pts3d[inl], pts2d[inl], self.camera.k, None, rvec, tvec)
        return PnpResult(SE3.from_rvec_tvec(rvec, tvec), inl.astype(np.intp), True)


class TrackingRun:
    def __init__(self, seq: RgbdSequence, frames: int, solver_cls: type[PnpSolver]) -> None:
        self.seq, self.n, self.solver_cls = seq, min(frames, len(seq)), solver_cls

    def run(self, label: str) -> None:
        pipe = VisualSlamPipeline(self.seq.camera, SlamConfig(loop_closure=False))
        old = pipe.odometry.pnp
        pipe.odometry.pnp = self.solver_cls(old.camera, old.reproj_error_px, old.iterations, old.min_inliers)
        ev, gt, est = TrajectoryEvaluator(), Trajectory.empty(), Trajectory.empty()
        jumps: list[str] = []
        prev: tuple[SE3, SE3] | None = None
        for i in range(self.n):
            f = self.seq.frame(i)
            g = self.seq.ground_truth(i)
            r = pipe.process(f)
            if g is None:
                continue
            gt.add(f.stamp, g)
            est.add(f.stamp, r.pose)
            if prev is not None:
                step_err = np.linalg.norm((prev[0].inverse() @ g).translation - (prev[1].inverse() @ r.pose).translation)
                if step_err > 0.05 and len(jumps) < 5:
                    jumps.append(f"frame {i}: step error {100 * step_err:.0f} cm, matches {r.matches}, "
                                 f"inliers {r.inliers}, new keyframe {r.new_keyframe}")
            prev = (g, r.pose)
            if (i + 1) % 100 == 0 or i == self.n - 1:
                rep = ev.evaluate(gt, est)
                ate = f"{100 * rep.ate_rmse_m:.1f} cm" if rep else "n/a"
                print(f"  [{label}] frame {i + 1}: ATE {ate}, keyframes {len(pipe.map.keyframes)}, state {r.state.value}")
        for j in jumps:
            print(f"  [{label}] {j}")


class CheckApp:
    @staticmethod
    def main() -> None:
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("--tum", default=str(Path.home() / "datasets" / "rgbd_dataset_freiburg1_xyz"))
        ap.add_argument("--tum-frames", type=int, default=400)
        a = ap.parse_args()
        print(f"numpy {np.__version__}, opencv {cv2.__version__}")
        sources: list[tuple[str, RgbdSequence, int]] = [("synthetic", SyntheticSource(), 10_000)]
        if Path(a.tum).expanduser().is_dir():
            sources.append(("tum", TumSequence(a.tum), a.tum_frames))
        for name, seq, n in sources:
            print(f"\n== {name}: {seq.name}")
            for label, cls in (("repo solver", PnpSolver), ("previous solver", PreviousPnpSolver)):
                TrackingRun(seq, n, cls).run(label)


if __name__ == "__main__":
    CheckApp.main()
