"""
localizer_node — grid localization or (ML-augmented) MCL as a ROS 2 localizer.

  method := grid | mcl | mcl_ml

  sub  /scan                sensor_msgs/LaserScan
       /odom                nav_msgs/Odometry         (wheel odometry from W1's sim / Gazebo)
       /initialpose         geometry_msgs/PoseWithCovarianceStamped   (RViz "2D Pose Estimate")
  pub  TF  map -> odom      the localizer's job (REP-105): corrects odometry drift
       /localization/pose   geometry_msgs/PoseWithCovarianceStamped  (same type as Nav2 AMCL's /amcl_pose)
       /localization/particles  geometry_msgs/PoseArray            (MCL)
       /localization/belief     nav_msgs/OccupancyGrid             (grid: belief marginal, 0..100)
       /localization/path       nav_msgs/Path
       /diagnostics
  srv  ~/global_localization (std_srvs/Trigger) — "I am lost": spread the belief over the whole map
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as PathMsg
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from sensor_motion_lab.core import Angle, GridRasterizer, Pose2D, Scan
from sensor_motion_lab.ros.common import LabNode, PathAccumulator, QoS, RosConv

from ..core.common import Localizer, PoseEstimate
from ..core.grid_localization import GridConfig, GridLocalizer
from ..core.learned import ScanPoseRegressor
from ..core.mcl import MclConfig, MonteCarloLocalizer
from .conv import CovarianceConv

DEFAULT_MODEL = str(Path.home() / "w2_models" / "pose_regressor.npz")


class LocalizerFactory:
    """`method` string -> configured localizer (Open/Closed)."""

    @staticmethod
    def create(method: str, node: LabNode) -> Localizer:
        grid = GridRasterizer(0.05).rasterize(node.world)
        rmax = node.robot.lidar.range_max
        if method == "grid":
            return GridLocalizer(grid, rmax, GridConfig(resolution=node.param("grid_resolution", 0.1),
                                                        heading_bins=node.param("heading_bins", 36)))
        cfg = MclConfig(num_particles=node.param("num_particles", 2000),
                        alpha_slow=node.param("recovery_alpha_slow", 0.001),
                        alpha_fast=node.param("recovery_alpha_fast", 0.1))
        if method == "mcl":
            return MonteCarloLocalizer(grid, rmax, cfg)
        if method == "mcl_ml":
            path = node.param("model_path", DEFAULT_MODEL)
            if not Path(path).expanduser().is_file():
                raise FileNotFoundError(f"no model at {path}: run `ros2 run localization_lab train_pose_regressor`")
            return MonteCarloLocalizer(grid, rmax, cfg, proposal=ScanPoseRegressor.load(path))
        raise ValueError(f"unknown method '{method}' (grid | mcl | mcl_ml)")


class LocalizerNode(LabNode):
    def __init__(self) -> None:
        super().__init__("localizer_node")
        self.method = self.param("method", "mcl")
        self.loc = LocalizerFactory.create(self.method, self)
        self.publish_tf = self.param("publish_tf", True)
        self.update_min_d = self.param("update_min_d", 0.1)  # same idea as AMCL's update_min_d / update_min_a
        self.update_min_a = self.param("update_min_a", 0.15)
        self.tf_period = 1.0 / self.param("tf_rate_hz", 20.0)

        self.odom: Pose2D | None = None
        self.last_update_odom: Pose2D | None = None
        self.est: PoseEstimate | None = None
        self.T_map_odom = Pose2D()
        self.needs_global_seed = self.param("init", "global") == "global"
        if not self.needs_global_seed:
            self.loc.reset_at(Pose2D(), 0.1, 0.1)  # the robot spawns at the map origin

        self.create_subscription(Odometry, "/odom", self._on_odom, 50)
        self.create_subscription(LaserScan, "/scan", self._on_scan, QoS.sensor())
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self._on_initialpose, 10)
        self.create_service(Trigger, "~/global_localization", self._on_global)
        self.tf = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/localization/pose", 10)
        self.particles_pub = self.create_publisher(PoseArray, "/localization/particles", 10)
        self.belief_pub = self.create_publisher(OccupancyGrid, "/localization/belief", 10)
        self.path_pub = self.create_publisher(PathMsg, "/localization/path", 10)
        self.path = PathAccumulator(self.MAP_FRAME)
        self.create_timer(self.tf_period, self._publish_tf)
        self.create_timer(self.param("report_period", 5.0), self._report)
        self.get_logger().info(f"Localizer '{self.loc.name}' ready (init={'global' if self.needs_global_seed else 'origin'})")

    # ----- inputs ----------------------------------------------------------- #
    def _on_odom(self, msg: Odometry) -> None:
        self.odom = self.odom_pose(msg)

    def _on_initialpose(self, msg: PoseWithCovarianceStamped) -> None:
        p = RosConv.pose_from_msg(msg.pose.pose)
        self.loc.reset_at(p, 0.25, 0.2)
        self.needs_global_seed = False
        self.get_logger().info(f"Initial pose from RViz: {p}")

    def _on_global(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        self.loc.reset_global()
        self.needs_global_seed = isinstance(self.loc, MonteCarloLocalizer) and self.loc.proposal is not None
        resp.success, resp.message = True, f"{self.loc.name}: belief spread over the whole map"
        return resp

    def _on_scan(self, msg: LaserScan) -> None:
        if self.odom is None:
            return
        scan = Scan.from_raw(np.asarray(msg.ranges), msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max)
        if self.needs_global_seed and isinstance(self.loc, MonteCarloLocalizer):
            self.loc.global_localize_with_proposal(scan)  # ML one-shot seed (no-op without a model)
            self.needs_global_seed = False
        prev = self.last_update_odom or self.odom
        moved = prev.distance_to(self.odom) >= self.update_min_d or \
            abs(Angle.wrap(self.odom.theta - prev.theta)) >= self.update_min_a
        if self.est is not None and not moved:
            return
        self.est = self.loc.update(prev, self.odom, scan)
        self.last_update_odom = self.odom
        # map -> odom = (map -> base) ∘ (odom -> base)^-1
        self.T_map_odom = self.est.pose.compose(self.odom.inverse())
        self._publish_estimate(msg.header.stamp)

    # ----- outputs ------------------------------------------------------------ #
    def _publish_tf(self) -> None:
        if self.publish_tf and self.est is not None:
            self.tf.sendTransform(RosConv.transform(self.T_map_odom, self.MAP_FRAME, self.ODOM_FRAME, self.now_msg()))

    def _publish_estimate(self, stamp) -> None:  # noqa: ANN001
        assert self.est is not None
        msg = PoseWithCovarianceStamped()
        msg.header.stamp, msg.header.frame_id = stamp, self.MAP_FRAME
        msg.pose.pose = RosConv.pose_to_msg(self.est.pose)
        msg.pose.covariance = CovarianceConv.to_ros(self.est.covariance)
        self.pose_pub.publish(msg)
        self.path.add(self.est.pose, stamp)
        self.path_pub.publish(self.path.msg)
        if isinstance(self.loc, MonteCarloLocalizer):
            pa = PoseArray()
            pa.header.stamp, pa.header.frame_id = stamp, self.MAP_FRAME
            step = max(1, len(self.loc.particles) // self.param("max_displayed_particles", 1000))
            pa.poses = [RosConv.pose_to_msg(Pose2D(*p)) for p in self.loc.particles[::step]]
            self.particles_pub.publish(pa)
        elif isinstance(self.loc, GridLocalizer):
            b = self.loc.belief_xy()
            x0, y0, res, _w, _h = self.loc.geometry
            self.belief_pub.publish(RosConv.occupancy_grid(np.round(100 * b / max(b.max(), 1e-12)), res, (x0, y0),
                                                           self.MAP_FRAME, stamp))

    def _report(self) -> None:
        if self.est is None:
            return
        m = {"method": self.loc.name, "update_ms": self.loc.stats.total_ms,
             "predict_ms": self.loc.stats.predict_ms, "correct_ms": self.loc.stats.correct_ms,
             "position_std_m": self.est.position_std, "heading_std_deg": self.est.heading_std_deg}
        if isinstance(self.loc, MonteCarloLocalizer):
            m |= {"particles": len(self.loc.particles), "n_eff": self.loc.n_eff, "injected": self.loc.injected}
        if isinstance(self.loc, GridLocalizer):
            m |= {"cells": self.loc.num_cells}
        self.diagnostics.publish(m, message="position_std shrinks as the localizer converges")
