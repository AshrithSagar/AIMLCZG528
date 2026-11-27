"""
sensor_model_node — evaluates a range-sensor model on the live /scan.

Every `eval_period` seconds, at the ground-truth pose matching the scan time:

 1. Sensor characterisation (forward model check)
      expected z* (ray cast)  ->  /sensor_model/expected_scan
      z − z* statistics vs. the ROBOTIS LDS-01 datasheet  ->  /diagnostics
 2. Generative use of the beam model
      z ~ p(z | z*) with realistic mixture weights  ->  /sensor_model/sampled_scan
 3. Inverse use (what a localiser does)
      log p(z | x, m) on a window of candidate poses around the truth
      ->  /sensor_model/likelihood (OccupancyGrid heat-map, 0..100)
      peak error / sharpness / compute time  ->  /diagnostics

  model := beam | likelihood_field   (Nav2 AMCL's two `laser_model_type`s)
         | correlation | feature     (map matching; line features + data association)
"""

from __future__ import annotations

import time

import numpy as np
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan

from ..core import (
    BeamModelParams,
    BeamRangeFinderModel,
    GridRasterizer,
    LidarSpec,
    LikelihoodFieldParams,
    LikelihoodSurfaceAnalyzer,
    LikelihoodSurfaceReport,
    Pose2D,
    RangeErrorAnalyzer,
    RangeSensorModelFactory,
    RayCaster,
    Scan,
)
from .common import LabNode, PoseBuffer, QoS, RosConv


class SensorModelNode(LabNode):
    def __init__(self) -> None:
        super().__init__("sensor_model_node")
        li = self.robot.lidar
        self.grid = GridRasterizer(self.param("resolution", 0.05)).rasterize(self.world)
        self.caster = RayCaster(self.grid, li.range_max)
        self.range_max = li.range_max

        beam = BeamModelParams(
            z_hit=self.param("z_hit", 0.5), z_short=self.param("z_short", 0.05),
            z_max=self.param("z_max", 0.05), z_rand=self.param("z_rand", 0.5),
            sigma_hit=self.param("sigma_hit", 0.2), lambda_short=self.param("lambda_short", 0.1),
        )
        lf = LikelihoodFieldParams(beam.z_hit, beam.z_rand, beam.sigma_hit,
                                   self.param("laser_likelihood_max_dist", 2.0))
        self.model = RangeSensorModelFactory.create(
            self.param("model", "likelihood_field"), self.grid, beam, lf, li.range_max,
            self.param("max_beams", 60), world=self.world,
        )
        self.generator = BeamRangeFinderModel(self.grid, BeamModelParams.realistic_lidar(), li.range_max)
        self.analyzer = RangeErrorAnalyzer(LidarSpec.lds01())
        self.rng = np.random.default_rng(self.param("seed", 0))
        self.half = self.param("window_half_size", 0.5)
        self.step = self.param("window_resolution", 0.025)

        self.truth = PoseBuffer()
        self.scan: LaserScan | None = None
        self.create_subscription(Odometry, "/ground_truth/odom", self._on_truth, 50)
        self.create_subscription(LaserScan, "/scan", self._on_scan, QoS.sensor())
        self.expected_pub = self.create_publisher(LaserScan, "/sensor_model/expected_scan", 10)
        self.sampled_pub = self.create_publisher(LaserScan, "/sensor_model/sampled_scan", 10)
        self.lik_pub = self.create_publisher(OccupancyGrid, "/sensor_model/likelihood", 10)
        self.create_timer(self.param("eval_period", 1.0), self._evaluate)
        self.get_logger().info(f"Sensor model: {self.model.name}, max_beams={self.model.max_beams}")

    def _on_truth(self, msg: Odometry) -> None:
        self.truth.add(RosConv.stamp_to_sec(msg.header.stamp), self.odom_pose(msg))

    def _on_scan(self, msg: LaserScan) -> None:
        self.scan = msg

    # ------------------------------------------------------------------ #
    def _evaluate(self) -> None:
        if self.scan is None:
            return
        msg = self.scan
        pose = self.truth.closest(RosConv.stamp_to_sec(msg.header.stamp))
        if pose is None:
            return
        scan = Scan.from_raw(np.asarray(msg.ranges), msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max)
        z_star = self.caster.cast(pose.as_array()[None, :], scan.angles)[0]
        self.expected_pub.publish(self._like(msg, z_star))
        self.sampled_pub.publish(self._like(msg, self.generator.sample(z_star, self.rng)))

        range_rep = self.analyzer.analyze(scan.ranges, z_star)
        surface_rep = self._likelihood_surface(scan, pose, msg)
        self.diagnostics.publish(
            {f"range_{k}": v for k, v in range_rep.as_dict().items()}
            | {f"{self.model.name}_{k}": v for k, v in surface_rep.as_dict().items()},
            message="range_* vs LDS-01 datasheet; *_peak_error_m = map-matching error",
        )

    def _likelihood_surface(self, scan: Scan, pose: Pose2D, msg: LaserScan) -> LikelihoodSurfaceReport:
        offs = np.arange(-self.half, self.half + 1e-9, self.step)
        gx, gy = np.meshgrid(pose.x + offs, pose.y + offs)
        cand = np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, pose.theta)))
        t0 = time.perf_counter()
        ll = self.model.log_likelihood(scan, cand)
        ms = 1000.0 * (time.perf_counter() - t0)
        img = np.exp(ll - ll.max()).reshape(gx.shape)
        origin = (float(gx.min() - self.step / 2), float(gy.min() - self.step / 2))
        self.lik_pub.publish(RosConv.occupancy_grid(np.round(100 * img), self.step, origin,
                                                    self.MAP_FRAME, msg.header.stamp))
        return LikelihoodSurfaceAnalyzer.analyze(cand[:, :2], ll, (pose.x, pose.y), ms)

    @staticmethod
    def _like(template: LaserScan, ranges: np.ndarray) -> LaserScan:
        out = LaserScan()
        out.header = template.header
        out.angle_min, out.angle_max, out.angle_increment = template.angle_min, template.angle_max, template.angle_increment
        out.time_increment, out.scan_time = template.time_increment, template.scan_time
        out.range_min, out.range_max = template.range_min, template.range_max
        out.ranges = [float(r) if r < template.range_max - 1e-3 else float("inf") for r in ranges]
        return out
