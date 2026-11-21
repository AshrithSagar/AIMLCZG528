"""
trajectory_overlay_node — show (and score) a trajectory computed *outside* ROS.

End-to-end deep SLAM systems (DPVO, DROID-SLAM) run as their own Python
programs on a GPU and write a TUM trajectory file. This node brings that result
into the ROS world, next to our SLAM, in the same `map` frame:

  param trajectory_file    TUM file (timestamp tx ty tz qx qy qz qw) — e.g. DPVO's saved trajectory
        sequence_path      the TUM sequence folder (its groundtruth.txt is the alignment target)
        sim3               true for monocular methods (scale is unobservable -> align with scale)
        stride             frame stride of the DL run; used only if the file is stamped with frame indices
                           (DPVO's demo.py: default --stride=2), mapped via the sequence's rgb.txt
  pub   /<label>/path      nav_msgs/Path (latched), aligned to the ground truth
        /diagnostics       ATE / RPE with the same evaluator as everything else
"""

from __future__ import annotations

from nav_msgs.msg import Path

from ..core.datasets import TumSequence
from ..core.metrics import FrameIndexStamps, Trajectory, TrajectoryEvaluator
from .common import Frames, MapConvention, QoS, RosConv, VslamNodeBase


class TrajectoryOverlayNode(VslamNodeBase):
    def __init__(self) -> None:
        super().__init__("trajectory_overlay_node")
        label = self.param("label", "dl_slam")
        evaluator = TrajectoryEvaluator(max_dt=self.param("max_dt", 0.02), with_scale=self.param("sim3", True))
        est = Trajectory.read_tum(self.param("trajectory_file", ""))
        seq = TumSequence(self.param("sequence_path", ""))
        if FrameIndexStamps.looks_like_indices(est):
            est = FrameIndexStamps(seq.root / "rgb.txt", self.param("stride", 2)).apply(est)
            self.get_logger().info("trajectory is stamped with frame indices: mapped to rgb.txt times")
        first = seq.ground_truth(0)
        if first is None:
            raise ValueError(f"{seq.root}: no ground truth for the first frame")
        # the same re-based world the player publishes its ground truth in (first frame = origin)
        gt = Trajectory([float(t) for t in seq.gt_stamps], [first.inverse() @ p for p in seq.gt_poses])
        aligned = evaluator.align(gt, est)  # raises if the timestamps do not overlap
        report = evaluator.evaluate(gt, est)
        assert report is not None

        stamp = self.get_clock().now().to_msg()
        self.pub = self.create_publisher(Path, f"/{label}/path", QoS.latched())
        self.pub.publish(RosConv.path([MapConvention.pose(p) for p in aligned.poses], [stamp] * len(aligned),
                                      Frames.MAP))
        self.metrics = {"label": label, "poses": len(est), **report.as_dict()}
        self.create_timer(self.param("report_period", 10.0), self._report)
        self._report()

    def _report(self) -> None:
        self.diagnostics.publish(self.metrics, message="external trajectory, TUM ATE/RPE (Sim(3) if monocular)")
