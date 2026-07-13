#!/usr/bin/env python3
"""
city_sim_node
=============
ROS 2 port of the ``CityBuilder`` / ``AutonomousVehicle`` / ``CityNavigationSLAM``
classes from *Lab_sheet_4-CNN_for_VisualSLAM.ipynb*.

Runs a headless (DIRECT-mode) PyBullet city with buildings, roads and four
distinctive landmarks (Tower / Dome / Monument / Fountain), drives a vehicle
around a fixed waypoint tour, and publishes:

  * ``/camera/image_raw``       (sensor_msgs/Image)   -- forward-facing camera
  * ``/camera/camera_info``     (sensor_msgs/CameraInfo)
  * ``/ground_truth/odom``      (nav_msgs/Odometry)   -- exact PyBullet pose
  * ``/ground_truth/path``      (nav_msgs/Path)
  * ``tf: world -> base_link_gt``

The CNN/ORB feature front end and pose estimation live in a separate node
(``cnn_feature_node``) so that the two responsibilities -- simulation and
perception -- stay decoupled, the way a real robot stack is organised.
"""

import time
from collections import deque

import numpy as np
import pybullet as p
import pybullet_data
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler


class CityBuilder:
    """Builds the static city: ground, roads, buildings and landmarks."""

    def __init__(self):
        self.buildings = []
        self.landmarks = []

    def build(self):
        ground = p.createCollisionShape(p.GEOM_BOX, halfExtents=[50, 50, 0.1])
        ground_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[50, 50, 0.1], rgbaColor=[0.3, 0.3, 0.3, 1]
        )
        p.createMultiBody(0, ground, ground_visual, [0, 0, -0.1])

        self._create_roads()
        self._create_buildings()
        self._create_landmarks()

    def _create_roads(self):
        road_color = [0.2, 0.2, 0.2, 1]
        road_width = 2.0
        for y in [-15, -5, 5, 15]:
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[25, road_width / 2, 0.05], rgbaColor=road_color
            )
            p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=[0, y, 0.05])
        for x in [-15, -5, 5, 15]:
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[road_width / 2, 25, 0.05], rgbaColor=road_color
            )
            p.createMultiBody(0, baseVisualShapeIndex=vis, basePosition=[x, 0, 0.05])

    def _create_buildings(self):
        configs = [
            (10, -20, 6, 12, [0.2, 0.2, 0.4, 1]),
            (10, -10, 5, 10, [0.3, 0.3, 0.5, 1]),
            (10, 10, 4, 8, [0.25, 0.25, 0.45, 1]),
            (-10, -20, 5, 10, [0.3, 0.3, 0.5, 1]),
            (0, -20, 4, 8, [0.5, 0.5, 0.6, 1]),
            (20, -20, 4, 6, [0.5, 0.5, 0.5, 1]),
            (20, -10, 3, 5, [0.6, 0.6, 0.6, 1]),
            (20, 10, 5, 9, [0.3, 0.4, 0.5, 1]),
            (-20, -20, 4, 6, [0.6, 0.4, 0.2, 1]),
            (-20, -10, 3, 5, [0.7, 0.5, 0.3, 1]),
            (-20, 0, 3, 5, [0.7, 0.5, 0.3, 1]),
            (-20, 10, 4, 7, [0.4, 0.4, 0.6, 1]),
            (-20, 20, 3, 4, [0.6, 0.6, 0.4, 1]),
            (-10, -10, 4, 6, [0.5, 0.4, 0.3, 1]),
            (-10, 10, 4, 5, [0.6, 0.5, 0.4, 1]),
            (-10, 20, 3, 7, [0.4, 0.5, 0.5, 1]),
            (0, 10, 5, 6, [0.6, 0.4, 0.4, 1]),
            (0, 20, 4, 5, [0.5, 0.6, 0.5, 1]),
            (10, 20, 5, 8, [0.4, 0.4, 0.5, 1]),
            (20, 0, 4, 7, [0.4, 0.5, 0.6, 1]),
            (20, 20, 3, 4, [0.7, 0.5, 0.4, 1]),
        ]
        for x, y, size, height, color in configs:
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[size / 2, size / 2, height / 2]
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[size / 2, size / 2, height / 2],
                rgbaColor=color,
            )
            p.createMultiBody(0, col, vis, [x, y, height / 2])
            self.buildings.append({"pos": [x, y], "size": size, "height": height})

    def _create_landmarks(self):
        tower_x, tower_y = -8, -8
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=1.5, height=15)
        vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=1.5, length=15, rgbaColor=[0.8, 0.2, 0.2, 1]
        )
        p.createMultiBody(0, col, vis, [tower_x, tower_y, 7.5])
        self.landmarks.append({"type": "Tower", "pos": [tower_x, tower_y]})

        dome_x, dome_y = 8, 8
        base_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[3, 3, 2], rgbaColor=[0.9, 0.9, 0.7, 1]
        )
        p.createMultiBody(
            0, baseVisualShapeIndex=base_vis, basePosition=[dome_x, dome_y, 2]
        )
        dome_col = p.createCollisionShape(p.GEOM_SPHERE, radius=2.5)
        dome_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=2.5, rgbaColor=[0.2, 0.7, 0.3, 1]
        )
        p.createMultiBody(0, dome_col, dome_vis, [dome_x, dome_y, 5.5])
        self.landmarks.append({"type": "Dome", "pos": [dome_x, dome_y]})

        mon_x, mon_y = 0, -8
        mon_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.8, 0.8, 10])
        mon_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.8, 0.8, 10], rgbaColor=[0.8, 0.8, 0.9, 1]
        )
        p.createMultiBody(0, mon_col, mon_vis, [mon_x, mon_y, 10])
        self.landmarks.append({"type": "Monument", "pos": [mon_x, mon_y]})

        fount_x, fount_y = -15, 12
        fount_vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=2, length=1, rgbaColor=[0.2, 0.4, 0.8, 1]
        )
        p.createMultiBody(
            0, baseVisualShapeIndex=fount_vis, basePosition=[fount_x, fount_y, 0.5]
        )
        self.landmarks.append({"type": "Fountain", "pos": [fount_x, fount_y]})


class AutonomousVehicle:
    """Simple kinematic vehicle with a forward-facing camera."""

    def __init__(self, start_xy):
        body_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.0, 0.5, 0.3])
        body_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[1.0, 0.5, 0.3], rgbaColor=[0.9, 0.1, 0.1, 1]
        )
        self.vehicle = p.createMultiBody(
            10, body_col, body_vis, [start_xy[0], start_xy[1], 0.5]
        )

        self.cam_width = 320
        self.cam_height = 240
        self.hfov_deg = 60.0

        self.waypoints = deque()

    def get_camera_image(self):
        pos, orn = p.getBasePositionAndOrientation(self.vehicle)
        rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

        cam_offset = np.array([0.5, 0, 0.5])
        cam_pos = pos + rot_mat @ cam_offset
        target = cam_pos + rot_mat @ np.array([5, 0, 0])
        up = rot_mat @ np.array([0, 0, 1])

        view_mat = p.computeViewMatrix(cam_pos, target, up)
        proj_mat = p.computeProjectionMatrixFOV(
            self.hfov_deg, self.cam_width / self.cam_height, 0.1, 100
        )

        _, _, rgb, _, _ = p.getCameraImage(
            self.cam_width,
            self.cam_height,
            view_mat,
            proj_mat,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb_array = np.array(rgb, dtype=np.uint8).reshape(
            self.cam_height, self.cam_width, 4
        )
        return rgb_array[:, :, :3]

    def move(self, velocity, steering, dt):
        pos, orn = p.getBasePositionAndOrientation(self.vehicle)
        euler = p.getEulerFromQuaternion(orn)

        new_yaw = euler[2] + steering * 0.1
        new_orn = p.getQuaternionFromEuler([0, 0, new_yaw])

        dx = velocity * np.cos(new_yaw) * dt
        dy = velocity * np.sin(new_yaw) * dt
        new_pos = [pos[0] + dx, pos[1] + dy, pos[2]]

        p.resetBasePositionAndOrientation(self.vehicle, new_pos, new_orn)

    def get_position(self):
        pos, orn = p.getBasePositionAndOrientation(self.vehicle)
        euler = p.getEulerFromQuaternion(orn)
        return np.array([pos[0], pos[1], euler[2]])

    def navigate_to_waypoint(self):
        if len(self.waypoints) == 0:
            return 0.0, 0.0, True

        target = self.waypoints[0]
        pos = self.get_position()
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        distance = np.hypot(dx, dy)

        if distance < 1.0:
            self.waypoints.popleft()
            if len(self.waypoints) == 0:
                return 0.0, 0.0, True

        desired_angle = np.arctan2(dy, dx)
        angle_diff = desired_angle - pos[2]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        velocity = min(1.0, distance / 2.0)
        steering = np.clip(angle_diff * 2.0, -1.0, 1.0)
        return velocity, steering, False


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


class CitySimNode(Node):
    def __init__(self):
        super().__init__("city_sim_node")

        self.declare_parameter("sim_rate_hz", 10.0)
        self.declare_parameter("loop_tour", True)
        rate_hz = self.get_parameter("sim_rate_hz").value
        self.loop_tour = self.get_parameter("loop_tour").value

        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        self.city = CityBuilder()
        self.city.build()
        self.vehicle = AutonomousVehicle(start_xy=[0, -18])
        self.vehicle.waypoints = deque(TOUR_WAYPOINTS)

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", 10)
        self.gt_odom_pub = self.create_publisher(Odometry, "/ground_truth/odom", 10)
        self.gt_path_pub = self.create_publisher(Path, "/ground_truth/path", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.gt_path = Path()
        self.gt_path.header.frame_id = "world"

        self.camera_info_msg = self._build_camera_info()

        self.dt = 1.0 / rate_hz
        self.timer = self.create_timer(self.dt, self.step)
        self.get_logger().info(
            f"City simulator ready: {len(self.city.buildings)} buildings, "
            f"{len(self.city.landmarks)} landmarks. Landmarks: "
            + ", ".join(lm["type"] for lm in self.city.landmarks)
        )

    def _build_camera_info(self):
        w, h = self.vehicle.cam_width, self.vehicle.cam_height
        fx = fy = (w / 2.0) / np.tan(np.radians(self.vehicle.hfov_deg) / 2.0)
        cx, cy = w / 2.0, h / 2.0
        msg = CameraInfo()
        msg.width, msg.height = w, h
        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        return msg

    def step(self):
        vel, steer, done = self.vehicle.navigate_to_waypoint()
        if done:
            if self.loop_tour:
                self.vehicle.waypoints = deque(TOUR_WAYPOINTS)
                vel, steer, done = self.vehicle.navigate_to_waypoint()
            else:
                return

        self.vehicle.move(vel, steer, self.dt)
        p.stepSimulation()

        now = self.get_clock().now().to_msg()

        cam_img = self.vehicle.get_camera_image()
        img_msg = self.bridge.cv2_to_imgmsg(cam_img, encoding="rgb8")
        img_msg.header.stamp = now
        img_msg.header.frame_id = "camera_link"
        self.image_pub.publish(img_msg)

        self.camera_info_msg.header.stamp = now
        self.camera_info_msg.header.frame_id = "camera_link"
        self.info_pub.publish(self.camera_info_msg)

        pos = self.vehicle.get_position()
        q = quaternion_from_euler(0, 0, pos[2])

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link_gt"
        odom.pose.pose.position.x = float(pos[0])
        odom.pose.pose.position.y = float(pos[1])
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        odom.twist.twist.linear.x = float(vel)
        odom.twist.twist.angular.z = float(steer)
        self.gt_odom_pub.publish(odom)

        pose_stamped = PoseStamped()
        pose_stamped.header = odom.header
        pose_stamped.pose = odom.pose.pose
        self.gt_path.header.stamp = now
        self.gt_path.poses.append(pose_stamped)
        self.gt_path_pub.publish(self.gt_path)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = now
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = "base_link_gt"
        tf_msg.transform.translation.x = float(pos[0])
        tf_msg.transform.translation.y = float(pos[1])
        tf_msg.transform.rotation.x = q[0]
        tf_msg.transform.rotation.y = q[1]
        tf_msg.transform.rotation.z = q[2]
        tf_msg.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(tf_msg)

    def destroy_node(self):
        p.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CitySimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
