"""
rig_driver_node — flies the Gazebo camera rig around the room in a circle
(constant speed + yaw rate = closed loop, so loop closure gets exercised).
The camera looks `yaw_offset_deg` outward from the direction of travel, so the
body-frame velocity has a sideways component (the rig "crabs" along the circle).

  pub  /rig/cmd_vel  geometry_msgs/Twist  (bridged to gz /model/camera_rig/cmd_vel)
"""

from __future__ import annotations

import math

from geometry_msgs.msg import Twist

from .common import VslamNodeBase


class RigDriverNode(VslamNodeBase):
    def __init__(self) -> None:
        super().__init__("rig_driver_node")
        self.speed = self.param("speed", 0.25)
        offset = math.radians(self.param("yaw_offset_deg", 35.0))
        self.vx, self.vy = self.speed * math.cos(offset), self.speed * math.sin(offset)
        radius = self.param("radius", 1.3)
        self.yaw_rate = self.speed / radius
        self.start_delay = self.param("start_delay", 3.0)
        self.started = self.get_clock().now()
        self.pub = self.create_publisher(Twist, "/rig/cmd_vel", 10)
        self.create_timer(0.05, self._tick)
        self.get_logger().info(f"Circling: v={self.speed} m/s, r={radius} m (one loop ≈ {6.283 * radius / self.speed:.0f} s)")

    def _tick(self) -> None:
        if (self.get_clock().now() - self.started).nanoseconds * 1e-9 < self.start_delay:
            return
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = float(self.vx), float(self.vy), float(self.yaw_rate)
        self.pub.publish(cmd)
