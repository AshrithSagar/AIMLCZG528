#!/usr/bin/env python3

from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler

TOUR_WAYPOINTS = [
    [0, -12],
    [10, -12],
    [10, -2],
    [15, -2],
    [15, 8],
    [8, 8],
    [0, 8],
    [0, 0],
    [-8, 0],
    [-8, -8],
    [0, -8],
    [0, -18],
]


class MockCitySimNode(Node):
    def __init__(self):
        super().__init__("city_sim_node")
        self.pose = np.array([0.0, -18.0, 0.0])  # x, y, yaw
        self.waypoints = deque(TOUR_WAYPOINTS)

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", 10)
        self.gt_odom_pub = self.create_publisher(Odometry, "/ground_truth/odom", 10)
        self.gt_path_pub = self.create_publisher(Path, "/ground_truth/path", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.gt_path = Path()
        self.gt_path.header.frame_id = "world"

        # Synthetic texture generation
        self.texture = (np.random.rand(600, 800, 3) * 255).astype(np.uint8)

        self.dt = 0.1
        self.timer = self.create_timer(self.dt, self.step)

    def step(self):
        if not self.waypoints:
            self.waypoints = deque(TOUR_WAYPOINTS)

        target = self.waypoints[0]
        dx, dy = target[0] - self.pose[0], target[1] - self.pose[1]
        distance = np.hypot(dx, dy)

        if distance < 1.0:
            self.waypoints.popleft()
            return

        desired_angle = np.arctan2(dy, dx)
        angle_diff = np.arctan2(
            np.sin(desired_angle - self.pose[2]), np.cos(desired_angle - self.pose[2])
        )

        vel = float(min(1.0, distance / 2.0))
        steer = float(np.clip(angle_diff * 2.0, -1.0, 1.0))

        # Kinematic update
        self.pose[2] += steer * 0.1
        self.pose[0] += vel * np.cos(self.pose[2]) * self.dt
        self.pose[1] += vel * np.sin(self.pose[2]) * self.dt

        now = self.get_clock().now().to_msg()

        # Generate synthetic image by shifting regional slices of the texture canvas
        shift_x = int(self.pose[0] * 50) % 400
        shift_y = int(self.pose[1] * 50) % 200
        cam_frame = self.texture[
            shift_y : shift_y + 240, shift_x : shift_x + 320
        ].copy()
        cv2.putText(
            cam_frame,
            "MOCK VSLAM MODE",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        img_msg = self.bridge.cv2_to_imgmsg(cam_frame, encoding="rgb8")
        img_msg.header.stamp = now
        img_msg.header.frame_id = "camera_link"
        self.image_pub.publish(img_msg)

        # Publish Camera Info
        info_msg = CameraInfo()
        info_msg.header.stamp = now
        info_msg.header.frame_id = "camera_link"
        info_msg.width, info_msg.height = 320, 240
        info_msg.k = [277.0, 0.0, 160.0, 0.0, 277.0, 120.0, 0.0, 0.0, 1.0]
        self.info_pub.publish(info_msg)

        # Odom and Path publications
        q = quaternion_from_euler(0, 0, self.pose[2])
        odom = Odometry(
            header=Header(stamp=now, frame_id="world"), child_frame_id="base_link_gt"
        )
        odom.pose.pose.position.x, odom.pose.pose.position.y = (
            self.pose[0],
            self.pose[1],
        )
        (
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        ) = q[0], q[1], q[2], q[3]
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = vel, steer
        self.gt_odom_pub.publish(odom)

        pose_stamped = PoseStamped(header=odom.header, pose=odom.pose.pose)
        self.gt_path.poses.append(pose_stamped)
        self.gt_path_pub.publish(self.gt_path)


def main(args=None):
    rclpy.init(args=args)
    node = MockCitySimNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
