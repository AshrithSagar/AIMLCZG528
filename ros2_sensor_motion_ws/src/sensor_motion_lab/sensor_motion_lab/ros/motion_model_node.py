"""
motion_model_node — propagates a particle set with a *sampling* motion model
(no sensor correction: this is pure prediction, i.e. what AMCL does between scans).

  model := odometry   uses consecutive /odom poses      (Thrun Table 5.6, Nav2 AMCL style)
  model := velocity   uses /cmd_vel and elapsed time     (Thrun Table 5.3)

  sub  /odom, /cmd_vel, /ground_truth/odom
  pub  /motion_model/particles   geometry_msgs/PoseArray  (frame map)
       /motion_model/mean_path   nav_msgs/Path
       /diagnostics              mean NEES, 95 % coverage, spread
  srv  ~/reset                   std_srvs/Trigger  (re-seed particles at the true pose)
"""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import PoseArray, Twist
from nav_msgs.msg import Odometry, Path
from std_srvs.srv import Trigger

from ..core import (
    Angle,
    ConsistencyTracker,
    MotionNoise,
    OdometryControl,
    OdometryMotionModel,
    Pose2D,
    PoseArrayOps,
    VelocityControl,
    VelocityMotionModel,
)
from .common import LabNode, PathAccumulator, RosConv


class MotionModelNode(LabNode):
    def __init__(self) -> None:
        super().__init__("motion_model_node")
        self.model_kind: str = self.param("model", "odometry")
        self.n = self.param("num_particles", 500)
        self.update_min_d = self.param("update_min_d", 0.05)
        self.update_min_a = self.param("update_min_a", 0.05)
        noise = MotionNoise(*(self.param(f"alpha{i}", d) for i, d in
                              ((1, 0.2), (2, 0.2), (3, 0.2), (4, 0.2), (5, 0.0), (6, 0.0))))
        rng = np.random.default_rng(self.param("seed", 0))
        self.odom_model = OdometryMotionModel(noise, rng)
        self.vel_model = VelocityMotionModel(noise, rng)

        self.particles = np.zeros((self.n, 3))
        self.last_odom: Pose2D | None = None
        self.odom_now: Pose2D | None = None
        self.truth: Pose2D | None = None
        self.cmd = Twist()
        self.tracker = ConsistencyTracker()
        self.mean_path = PathAccumulator(self.MAP_FRAME)

        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(Odometry, "/ground_truth/odom", self._on_truth, 10)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)
        self.particles_pub = self.create_publisher(PoseArray, "/motion_model/particles", 10)
        self.path_pub = self.create_publisher(Path, "/motion_model/mean_path", 10)
        self.create_service(Trigger, "~/reset", self._on_reset)

        self.vel_dt = 1.0 / self.param("velocity_rate_hz", 10.0)
        if self.model_kind == "velocity":
            self.create_timer(self.vel_dt, self._velocity_step)
        self.create_timer(1.0 / self.param("publish_rate_hz", 5.0), self._publish)
        self.create_timer(self.param("report_period", 5.0), self._report)
        self.get_logger().info(f"Motion model '{self.model_kind}' with {noise}, N={self.n}")

    # ----- inputs ------------------------------------------------------ #
    def _on_truth(self, msg: Odometry) -> None:
        first = self.truth is None
        self.truth = self.odom_pose(msg)
        if first:
            self._seed_at(self.truth)

    def _on_cmd(self, msg: Twist) -> None:
        self.cmd = msg

    def _on_odom(self, msg: Odometry) -> None:
        self.odom_now = self.odom_pose(msg)
        if self.last_odom is None:
            self.last_odom = self.odom_now
            return
        if self.model_kind != "odometry":
            return
        moved = self.last_odom.distance_to(self.odom_now)
        turned = abs(Angle.wrap(self.odom_now.theta - self.last_odom.theta))
        if moved < self.update_min_d and turned < self.update_min_a:
            return  # same gating as AMCL's update_min_d / update_min_a
        self.particles = self.odom_model.sample(self.particles, OdometryControl(self.last_odom, self.odom_now))
        self.last_odom = self.odom_now
        self._after_update()

    def _velocity_step(self) -> None:
        v, w = self.cmd.linear.x, self.cmd.angular.z
        if abs(v) < 1e-4 and abs(w) < 1e-4:
            return
        self.particles = self.vel_model.sample(self.particles, VelocityControl(v, w, self.vel_dt))
        self._after_update()

    # ----- bookkeeping ------------------------------------------------- #
    def _after_update(self) -> None:
        self.mean_path.add(PoseArrayOps.mean(self.particles), self.now_msg())
        if self.truth is not None:
            self.tracker.update(self.particles, (self.truth.x, self.truth.y))

    def _seed_at(self, pose: Pose2D) -> None:
        self.particles = np.tile(pose.as_array(), (self.n, 1))
        self.last_odom = self.odom_now
        self.tracker = ConsistencyTracker()
        self.mean_path.clear()

    def _on_reset(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        if self.truth is None:
            resp.success, resp.message = False, "no ground truth yet"
        else:
            self._seed_at(self.truth)
            resp.success, resp.message = True, "particles re-seeded at the true pose"
        return resp

    # ----- outputs ----------------------------------------------------- #
    def _publish(self) -> None:
        msg = PoseArray()
        msg.header.frame_id, msg.header.stamp = self.MAP_FRAME, self.now_msg()
        msg.poses = [RosConv.pose_to_msg(Pose2D(*p)) for p in self.particles]
        self.particles_pub.publish(msg)
        self.path_pub.publish(self.mean_path.msg)

    def _report(self) -> None:
        if not self.tracker.nees:
            return
        cov = PoseArrayOps.position_covariance(self.particles)
        self.diagnostics.publish(
            {
                "model": self.model_kind,
                "mean_nees": self.tracker.mean_nees,
                "last_nees": self.tracker.nees[-1],
                "coverage_95": self.tracker.coverage_95,
                "pos_std_m": float(np.sqrt(np.trace(cov))),
                "heading_std_deg": float(np.degrees(np.std(self.particles[:, 2]))),
            },
            message="E[NEES]=2 and coverage≈0.95 mean the alphas are honest",
        )
