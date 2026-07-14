#!/usr/bin/env python3

from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from tf_transformations import euler_from_quaternion

TOUR_WAYPOINTS = [
    [4.0, 0.0],
    [4.0, 4.0],
    [-4.0, 4.0],
    [-4.0, -4.0],
    [4.0, -4.0],
    [0.0, 0.0],
]


class WaypointDriverNode(Node):
    def __init__(self):
        super().__init__("waypoint_driver_node")

        self.declare_parameter("control_rate_hz", 10.0)
        rate_hz = self.get_parameter("control_rate_hz").value

        self.waypoints = deque(TOUR_WAYPOINTS)
        self.pose = None

        # Subscribing to Ground Truth Path from the mock simulator
        self.path_sub = self.create_subscription(
            Path, "/ground_truth/path", self.on_path, 10
        )
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.timer = self.create_timer(1.0 / rate_hz, self.control_step)
        self.get_logger().info(
            "Waypoint driver ready, navigating via mock ground truth..."
        )

    def on_path(self, msg: Path):
        if not msg.poses:
            return
        # Extract the latest position to steer the robot
        latest_pose = msg.poses[-1].pose
        q = latest_pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.pose = np.array([latest_pose.position.x, latest_pose.position.y, yaw])

    def control_step(self):
        cmd = Twist()
        if self.pose is None:
            # Force a tiny forward movement initialization to jumpstart the loops
            cmd.linear.x = 0.1
            self.cmd_pub.publish(cmd)
            return

        if len(self.waypoints) == 0:
            self.waypoints = deque(TOUR_WAYPOINTS)

        target = self.waypoints[0]
        x, y, theta = self.pose
        dx, dy = target[0] - x, target[1] - y
        distance = np.hypot(dx, dy)

        if distance < 0.4:
            self.waypoints.popleft()
            self.cmd_pub.publish(cmd)
            return

        desired = np.arctan2(dy, dx)
        angle_diff = np.arctan2(np.sin(desired - theta), np.cos(desired - theta))

        cmd.linear.x = float(min(0.5, distance))
        cmd.angular.z = float(np.clip(angle_diff * 2.0, -1.0, 1.0))
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
