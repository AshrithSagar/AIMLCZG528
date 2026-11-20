"""
pose_regressor_node — the "replace it by ML" localizer: one MLP forward pass per scan.

  sub  /scan          sensor_msgs/LaserScan
  pub  /ml/pose       geometry_msgs/PoseWithCovarianceStamped   (map frame; no filter, no memory)
       /diagnostics   inference time

Train first:  ros2 run localization_lab train_pose_regressor
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan

from sensor_motion_lab.core import Scan
from sensor_motion_lab.ros.common import LabNode, QoS, RosConv

from ..core.learned import ScanPoseRegressor
from .conv import CovarianceConv
from .localizer_node import DEFAULT_MODEL


class PoseRegressorNode(LabNode):
    def __init__(self) -> None:
        super().__init__("pose_regressor_node")
        path = Path(self.param("model_path", DEFAULT_MODEL)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"no model at {path}: run `ros2 run localization_lab train_pose_regressor`")
        self.reg = ScanPoseRegressor.load(path)
        std = self.param("reported_std_xy", 0.3)  # ~ the validation median error
        self.cov = np.diag([std**2, std**2, self.param("reported_std_theta", 0.15) ** 2])
        self.pub = self.create_publisher(PoseWithCovarianceStamped, "/ml/pose", 10)
        self.create_subscription(LaserScan, "/scan", self._on_scan, QoS.sensor())
        self.times: list[float] = []
        self.create_timer(self.param("report_period", 5.0), self._report)
        self.get_logger().info(f"Pose regressor loaded from {path} ({self.reg.net.num_params} parameters)")

    def _on_scan(self, msg: LaserScan) -> None:
        scan = Scan.from_raw(np.asarray(msg.ranges), msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max)
        t0 = time.perf_counter()
        pose = self.reg.propose(scan)
        self.times.append(1000 * (time.perf_counter() - t0))
        out = PoseWithCovarianceStamped()
        out.header.stamp, out.header.frame_id = msg.header.stamp, self.MAP_FRAME
        out.pose.pose = RosConv.pose_to_msg(pose)
        out.pose.covariance = CovarianceConv.to_ros(self.cov)
        self.pub.publish(out)

    def _report(self) -> None:
        if self.times:
            self.diagnostics.publish({"inference_ms_mean": float(np.mean(self.times[-50:])),
                                      "inference_ms_max": float(np.max(self.times[-50:]))},
                                     message="CPU inference per scan", log=False)
