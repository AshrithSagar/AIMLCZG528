"""
trajectory_evaluator_node — ATE / RPE / drift of an estimate against ground truth.

  sub  /ground_truth/odom           (reference)
       <estimate_topic>             (default /odom = raw wheel odometry)
  pub  /ground_truth/path, <estimate_path_topic>   nav_msgs/Path for RViz
       /diagnostics                 periodic report
  srv  ~/report                     std_srvs/Trigger: compute now + write TUM files

The TUM files (`ground_truth.tum`, `estimate.tum`) can be fed straight into the
standard `evo` tool for an independent check:

    evo_ape tum ground_truth.tum estimate.tum --plot
    evo_rpe tum ground_truth.tum estimate.tum --delta 0.5 --delta_unit m
"""

from __future__ import annotations

from pathlib import Path

from nav_msgs.msg import Odometry, Path as PathMsg
from std_srvs.srv import Trigger

from ..core import Trajectory, TrajectoryEvaluator
from .common import LabNode, PathAccumulator, RosConv


class TrajectoryRecorder:
    """Accumulates (stamp, pose) samples for one trajectory."""

    def __init__(self) -> None:
        self.stamps: list[float] = []
        self.poses: list[tuple[float, float, float]] = []

    def add(self, t: float, x: float, y: float, th: float) -> None:
        if self.stamps and t <= self.stamps[-1]:
            return  # drop duplicates / out-of-order samples
        self.stamps.append(t)
        self.poses.append((x, y, th))

    def as_trajectory(self) -> Trajectory:
        return Trajectory.from_lists(self.stamps, self.poses)

    def __len__(self) -> int:
        return len(self.stamps)


class TrajectoryEvaluatorNode(LabNode):
    MIN_SAMPLES = 20

    def __init__(self) -> None:
        super().__init__("trajectory_evaluator_node")
        est_topic = self.param("estimate_topic", "/odom")
        est_path_topic = self.param("estimate_path_topic", "/odom_path")
        self.out_dir = Path(self.param("output_dir", str(Path.home() / "w1_results"))).expanduser()
        self.evaluator = TrajectoryEvaluator(align=self.param("align", False), rpe_delta_m=self.param("rpe_delta_m", 0.5))

        self.gt, self.est = TrajectoryRecorder(), TrajectoryRecorder()
        self.gt_path, self.est_path = PathAccumulator(self.MAP_FRAME), PathAccumulator(self.MAP_FRAME)
        self.create_subscription(Odometry, "/ground_truth/odom", self._on_gt, 50)
        self.create_subscription(Odometry, est_topic, self._on_est, 50)
        self.gt_path_pub = self.create_publisher(PathMsg, "/ground_truth/path", 10)
        self.est_path_pub = self.create_publisher(PathMsg, est_path_topic, 10)
        self.create_service(Trigger, "~/report", self._on_report)
        self.create_timer(0.5, self._publish_paths)
        self.create_timer(self.param("report_period", 10.0), self._report)
        self.get_logger().info(f"Evaluating '{est_topic}' against /ground_truth/odom")

    def _on_gt(self, msg: Odometry) -> None:
        self._record(msg, self.gt, self.gt_path)

    def _on_est(self, msg: Odometry) -> None:
        # Estimate is expressed in `odom`; map->odom is identity in this lab (robot starts at origin).
        self._record(msg, self.est, self.est_path)

    def _record(self, msg: Odometry, rec: TrajectoryRecorder, path: PathAccumulator) -> None:
        p = self.odom_pose(msg)
        rec.add(RosConv.stamp_to_sec(msg.header.stamp), p.x, p.y, p.theta)
        path.add(p, msg.header.stamp)

    def _publish_paths(self) -> None:
        self.gt_path_pub.publish(self.gt_path.msg)
        self.est_path_pub.publish(self.est_path.msg)

    # ------------------------------------------------------------------ #
    def _report(self, write_files: bool = False) -> str:
        if len(self.gt) < self.MIN_SAMPLES or len(self.est) < self.MIN_SAMPLES:
            return "not enough samples yet"
        gt, est = self.gt.as_trajectory(), self.est.as_trajectory()
        rep = self.evaluator.evaluate(gt, est)
        self.diagnostics.publish(rep.as_dict(), message="ATE/RPE per TUM RGB-D benchmark definitions")
        if write_files:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            gt.write_tum(self.out_dir / "ground_truth.tum")
            est.write_tum(self.out_dir / "estimate.tum")
            return f"ATE {rep.ate_rmse:.3f} m, drift {rep.drift_percent:.2f} %; TUM files in {self.out_dir}"
        return f"ATE {rep.ate_rmse:.3f} m"

    def _on_report(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        resp.message = self._report(write_files=True)
        resp.success = not resp.message.startswith("not enough")
        return resp

    def on_shutdown(self) -> None:
        try:
            self.get_logger().info(self._report(write_files=True))
        except Exception as exc:  # noqa: BLE001 - best effort at shutdown
            print(f"trajectory_evaluator_node: final report failed: {exc}")
