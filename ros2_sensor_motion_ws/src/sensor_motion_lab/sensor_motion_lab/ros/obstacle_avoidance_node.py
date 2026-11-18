"""
obstacle_avoidance_node — Lab Manual 2's experiment, as a proper ROS 2 node.

The manual's `moving_bot.py` did everything in one node (kinematics, fake
LiDAR, TF, control). Here the *controller* is the only thing this node does;
the simulator (kinematic or Gazebo) supplies /scan and executes /cmd_vel.

  sub  /scan     sensor_msgs/LaserScan
  pub  /cmd_vel  geometry_msgs/Twist
  param duration  > 0 stops the robot after that many seconds (Practice item 1)
"""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from ..core import AvoidanceConfig, ReactiveAvoidanceController
from .common import LabNode, QoS


class ObstacleAvoidanceNode(LabNode):
    def __init__(self) -> None:
        super().__init__("obstacle_avoidance_node")
        d = AvoidanceConfig()
        self.controller = ReactiveAvoidanceController(AvoidanceConfig(
            forward_speed=self.param("forward_speed", d.forward_speed),
            turn_rate=self.param("turn_rate", d.turn_rate),
            threshold=self.param("threshold", d.threshold),
            front_sector=np.radians(self.param("front_sector_deg", float(np.degrees(d.front_sector)))),
        ))
        self.duration = self.param("duration", 0.0)
        self.started = self.get_clock().now()
        self.stopped = False
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(LaserScan, "/scan", self._on_scan, QoS.sensor())

    def _on_scan(self, msg: LaserScan) -> None:
        elapsed = (self.get_clock().now() - self.started).nanoseconds * 1e-9
        if self.duration > 0.0 and elapsed >= self.duration:
            if not self.stopped:
                self.get_logger().info(f"duration {self.duration:.1f} s reached, stopping")
                self.stopped = True
            self._publish(0.0, 0.0)
            return
        ranges = np.asarray(msg.ranges, dtype=np.float64)
        angles = msg.angle_min + msg.angle_increment * np.arange(ranges.size)
        self._publish(*self.controller.command(ranges, angles))

    def _publish(self, v: float, w: float) -> None:
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(v), float(w)
        self.cmd_pub.publish(cmd)
