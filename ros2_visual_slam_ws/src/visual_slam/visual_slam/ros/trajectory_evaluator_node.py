"""
trajectory_evaluator_node — live ATE / RPE of any pose estimate vs. ground truth.

  sub  ground truth   /ground_truth/pose (PoseStamped, dataset/synthetic)  or
                      /ground_truth/odom (Odometry, Gazebo)               — `ground_truth_type`
       estimate       /vslam/odometry (Odometry)  — or any other SLAM's odometry, e.g. RTAB-Map's
       optimised      /vslam/optimized_path (Path) — loop-closure-corrected trajectory
  pub  /diagnostics   ATE/RPE (TUM benchmark definitions, SE(3)-aligned)
  srv  ~/report       std_srvs/Trigger: compute now + write TUM files for `evo`

Because the estimate is just a topic, the *same* node benchmarks our SLAM and
a reference package (RTAB-Map's `/rtabmap/odom`) on identical input.
"""

from __future__ import annotations

from pathlib import Path as FsPath

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_srvs.srv import Trigger

from ..core.metrics import Trajectory, TrajectoryEvaluator
from .common import RosConv, VslamNodeBase


class TrajectoryEvaluatorNode(VslamNodeBase):
    def __init__(self) -> None:
        super().__init__("trajectory_evaluator_node")
        gt_type = self.param("ground_truth_type", "pose")
        gt_topic = self.param("ground_truth_topic", "/ground_truth/pose" if gt_type == "pose" else "/ground_truth/odom")
        est_topic = self.param("estimate_topic", "/vslam/odometry")
        self.label = self.param("label", "vslam")
        self.out_dir = FsPath(self.param("output_dir", str(FsPath.home() / "vslam_results"))).expanduser()
        self.evaluator = TrajectoryEvaluator(max_dt=self.param("max_dt", 0.02))
        self.gt, self.est, self.opt = Trajectory.empty(), Trajectory.empty(), Trajectory.empty()

        if gt_type == "pose":
            self.create_subscription(PoseStamped, gt_topic, self._on_gt_pose, 50)
        else:
            self.create_subscription(Odometry, gt_topic, self._on_gt_odom, 50)
        self.create_subscription(Odometry, est_topic, self._on_est, 50)
        self.create_subscription(Path, self.param("optimized_path_topic", "/vslam/optimized_path"), self._on_opt, 1)
        self.create_service(Trigger, "~/report", self._on_report)
        self.create_timer(self.param("report_period", 10.0), self._report)
        self.get_logger().info(f"Evaluating '{est_topic}' against '{gt_topic}' ({gt_type})")

    def _on_gt_pose(self, msg: PoseStamped) -> None:
        self.gt.add(RosConv.stamp_to_sec(msg.header.stamp), RosConv.se3(msg.pose))

    def _on_gt_odom(self, msg: Odometry) -> None:
        self.gt.add(RosConv.stamp_to_sec(msg.header.stamp), RosConv.se3(msg.pose.pose))

    def _on_est(self, msg: Odometry) -> None:
        self.est.add(RosConv.stamp_to_sec(msg.header.stamp), RosConv.se3(msg.pose.pose))

    def _on_opt(self, msg: Path) -> None:
        traj = Trajectory.empty()
        for ps in msg.poses:
            traj.add(RosConv.stamp_to_sec(ps.header.stamp), RosConv.se3(ps.pose))
        self.opt = traj

    # ------------------------------------------------------------------ #
    def _report(self, write: bool = False) -> str:
        live = self.evaluator.evaluate(self.gt, self.est)
        if live is None:
            return "not enough associated poses yet (check that stamps match)"
        metrics = {f"live_{k}": v for k, v in live.as_dict().items()}
        opt = self.evaluator.evaluate(self.gt, self.opt) if len(self.opt) else None
        if opt is not None:
            metrics["optimized_ate_rmse_m"] = opt.ate_rmse_m
        self.diagnostics.publish(metrics, message=f"{self.label}: TUM ATE/RPE after SE(3) alignment")
        if write:
            self.gt.write_tum(self.out_dir / "ground_truth.tum")
            self.est.write_tum(self.out_dir / f"{self.label}_live.tum")
            if len(self.opt):
                self.opt.write_tum(self.out_dir / f"{self.label}_optimized.tum")
            return f"ATE live {live.ate_rmse_m * 100:.1f} cm" + \
                (f", optimised {opt.ate_rmse_m * 100:.1f} cm" if opt else "") + f"; TUM files in {self.out_dir}"
        return f"ATE {live.ate_rmse_m:.3f} m"

    def _on_report(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        resp.message = self._report(write=True)
        resp.success = not resp.message.startswith("not enough")
        return resp

    def on_shutdown(self) -> None:
        try:
            print(f"[trajectory_evaluator_node] {self._report(write=True)}")
        except Exception as exc:  # noqa: BLE001 - best effort at shutdown
            print(f"[trajectory_evaluator_node] final report failed: {exc}")
