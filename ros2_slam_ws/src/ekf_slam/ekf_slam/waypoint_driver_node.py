#!/usr/bin/env python3
"""
waypoint_driver_node
=====================
Replaces the PyBullet version's internal "move the robot" logic. In the
Gazebo build, the robot is a real (if simple) diff-drive model with its
own physics, so instead of teleporting it each tick we drive it with
``geometry_msgs/Twist`` commands on ``/cmd_vel``, exactly as you would a
real robot.

Subscribes:
  * ``/ground_truth/odom_raw`` (nav_msgs/Odometry) -- bridged from Gazebo's
    diff-drive plugin, used only to compute the next waypoint-following
    command (not fed to the EKF -- that gets its own noisy copy from
    ``landmark_sensor_node``).

Publishes:
  * ``/cmd_vel`` (geometry_msgs/Twist)
"""

from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf_transformations import euler_from_quaternion

TOUR_WAYPOINTS = [[4, 0], [4, 4], [-4, 4], [-4, -4], [4, -4], [0, 0]]


class WaypointDriverNode(Node):
    def __init__(self):
        super().__init__("waypoint_driver_node")

        self.declare_parameter("control_rate_hz", 10.0)
        rate_hz = self.get_parameter("control_rate_hz").value

        self.waypoints = deque(TOUR_WAYPOINTS)
        self.pose = None  # (x, y, theta), filled in on first odom message

        self.odom_sub = self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.timer = self.create_timer(1.0 / rate_hz, self.control_step)
        self.get_logger().info(
            "Waypoint driver ready, waiting for first odometry message..."
        )

    def on_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])

    def control_step(self):
        cmd = Twist()
        if self.pose is None:
            self.cmd_pub.publish(cmd)  # stay put until we have a pose
            return

        if len(self.waypoints) == 0:
            self.waypoints = deque(TOUR_WAYPOINTS)  # loop the tour forever

        target = self.waypoints[0]
        x, y, theta = self.pose
        dx, dy = target[0] - x, target[1] - y
        distance = np.hypot(dx, dy)

        if distance < 0.3:
            self.waypoints.popleft()
            self.cmd_pub.publish(cmd)
            return

        desired = np.arctan2(dy, dx)
        angle_diff = np.arctan2(np.sin(desired - theta), np.cos(desired - theta))

        cmd.linear.x = float(min(0.6, distance))
        cmd.angular.z = float(np.clip(angle_diff * 2.0, -1.5, 1.5))
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
