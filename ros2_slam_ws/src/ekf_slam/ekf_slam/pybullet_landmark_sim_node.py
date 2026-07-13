#!/usr/bin/env python3
"""
landmark_sim_node
==================
ROS 2 port of the environment half of the notebook's ``VisualSLAM_Colab``
class: a PyBullet world with a differential-drive-like robot and six
coloured point landmarks, driven around a fixed square tour.

Landmark *identity* here is treated as already resolved -- exactly as in
the notebook, where each landmark carries a colour/id and the sensor model
reports ``(id, range, bearing)`` directly. In a full pipeline that
identity would come from a classifier (e.g. the CNN front end in the
``cnn_visual_slam`` package) recognising each colour blob; we keep that
step external so this package can focus purely on the EKF-SLAM back end.

Publishes:
  * ``/odom``                    (nav_msgs/Odometry)  -- noisy wheel odometry
  * ``/landmark_observations``   (slam_sim_msgs/LandmarkObservationArray)
  * ``/ground_truth/path``       (nav_msgs/Path)
  * ``/ground_truth/landmarks``  (visualization_msgs/MarkerArray)
  * ``tf: world -> base_link_gt``
"""

from collections import deque

import numpy as np
import pybullet as p
import pybullet_data
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from slam_sim_msgs.msg import LandmarkObservation, LandmarkObservationArray
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray

LANDMARK_CONFIGS = [
    (0, [3, 3, 0.3], [1, 0, 0, 1], "Red"),
    (1, [3, -3, 0.3], [0, 1, 0, 1], "Green"),
    (2, [-3, 3, 0.3], [0, 0, 1, 1], "Blue"),
    (3, [-3, -3, 0.3], [1, 1, 0, 1], "Yellow"),
    (4, [0, 4, 0.3], [1, 0, 1, 1], "Magenta"),
    (5, [4, 0, 0.3], [0, 1, 1, 1], "Cyan"),
]

TOUR_WAYPOINTS = [[4, 0], [4, 4], [-4, 4], [-4, -4], [4, -4], [0, 0]]

SENSOR_RANGE_MAX = 6.0
SENSOR_FOV = np.pi  # +-90 deg either side of heading


class LandmarkSimNode(Node):
    def __init__(self):
        super().__init__("landmark_sim_node")

        self.declare_parameter("sim_rate_hz", 10.0)
        self.declare_parameter("odom_v_noise_std", 0.03)
        self.declare_parameter("odom_w_noise_std", 0.02)
        self.declare_parameter("range_noise_std", 0.15)
        self.declare_parameter("bearing_noise_std", 0.05)

        rate_hz = self.get_parameter("sim_rate_hz").value
        self.odom_v_noise_std = self.get_parameter("odom_v_noise_std").value
        self.odom_w_noise_std = self.get_parameter("odom_w_noise_std").value
        self.range_noise_std = self.get_parameter("range_noise_std").value
        self.bearing_noise_std = self.get_parameter("bearing_noise_std").value

        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.loadURDF("plane.urdf")

        self.robot = self._create_robot()
        self.landmark_positions = {}
        for lm_id, pos, color, name in LANDMARK_CONFIGS:
            self._create_landmark_visual(pos, color)
            self.landmark_positions[lm_id] = np.array(pos[:2])

        self.true_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.waypoints = deque(TOUR_WAYPOINTS)

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.obs_pub = self.create_publisher(
            LandmarkObservationArray, "/landmark_observations", 10
        )
        self.gt_path_pub = self.create_publisher(Path, "/ground_truth/path", 10)
        self.gt_landmarks_pub = self.create_publisher(
            MarkerArray, "/ground_truth/landmarks", 10
        )
        self.tf_broadcaster = TransformBroadcaster(self)

        self.gt_path = Path()
        self.gt_path.header.frame_id = "world"

        self.rng = np.random.default_rng(42)
        self.dt = 1.0 / rate_hz
        self.timer = self.create_timer(self.dt, self.step)
        self._publish_landmark_markers()
        self.get_logger().info(
            f"Landmark world ready with {len(self.landmark_positions)} landmarks."
        )

    def _create_robot(self):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.15, 0.1])
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.2, 0.15, 0.1], rgbaColor=[0.2, 0.2, 0.8, 1]
        )
        return p.createMultiBody(1.0, col, vis, [0, 0, 0.1])

    def _create_landmark_visual(self, pos, color):
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=color)
        p.createMultiBody(0, col, vis, pos)

    def _navigate_to_waypoint(self):
        if len(self.waypoints) == 0:
            self.waypoints = deque(TOUR_WAYPOINTS)

        target = self.waypoints[0]
        x, y, theta = self.true_pose
        dx, dy = target[0] - x, target[1] - y
        distance = np.hypot(dx, dy)
        if distance < 0.3:
            self.waypoints.popleft()
            return 0.0, 0.0

        desired = np.arctan2(dy, dx)
        angle_diff = np.arctan2(np.sin(desired - theta), np.cos(desired - theta))
        v = min(0.6, distance)
        w = np.clip(angle_diff * 2.0, -1.5, 1.5)
        return v, w

    def step(self):
        v, w = self._navigate_to_waypoint()

        x, y, theta = self.true_pose
        theta_new = theta + w * self.dt
        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        self.true_pose = np.array([x_new, y_new, theta_new])

        p.resetBasePositionAndOrientation(
            self.robot, [x_new, y_new, 0.1], p.getQuaternionFromEuler([0, 0, theta_new])
        )
        p.stepSimulation()

        # Noisy wheel-odometry estimate of the commanded velocities (this is what
        # the EKF's prediction step actually receives -- never the ground truth).
        v_meas = v + self.rng.normal(0, self.odom_v_noise_std)
        w_meas = w + self.rng.normal(0, self.odom_w_noise_std)

        now = self.get_clock().now().to_msg()
        self._publish_odom(v_meas, w_meas, now)
        self._publish_ground_truth(now)
        self._publish_observations(now)

    def _publish_odom(self, v_meas, w_meas, stamp):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.twist.twist.linear.x = float(v_meas)
        odom.twist.twist.angular.z = float(w_meas)
        self.odom_pub.publish(odom)

    def _publish_ground_truth(self, stamp):
        x, y, theta = self.true_pose
        q = quaternion_from_euler(0, 0, theta)

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = "world"
        pose_stamped.pose.position.x = float(x)
        pose_stamped.pose.position.y = float(y)
        pose_stamped.pose.orientation.x = q[0]
        pose_stamped.pose.orientation.y = q[1]
        pose_stamped.pose.orientation.z = q[2]
        pose_stamped.pose.orientation.w = q[3]
        self.gt_path.header.stamp = stamp
        self.gt_path.poses.append(pose_stamped)
        self.gt_path_pub.publish(self.gt_path)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = "base_link_gt"
        tf_msg.transform.translation.x = float(x)
        tf_msg.transform.translation.y = float(y)
        tf_msg.transform.rotation.x = q[0]
        tf_msg.transform.rotation.y = q[1]
        tf_msg.transform.rotation.z = q[2]
        tf_msg.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(tf_msg)

    def _publish_observations(self, stamp):
        x, y, theta = self.true_pose
        arr = LandmarkObservationArray()
        arr.header.stamp = stamp
        arr.header.frame_id = "base_link_gt"

        for lm_id, lm_pos in self.landmark_positions.items():
            dx, dy = lm_pos[0] - x, lm_pos[1] - y
            true_range = np.hypot(dx, dy)
            raw_bearing = np.arctan2(dy, dx) - theta
            true_bearing = np.arctan2(np.sin(raw_bearing), np.cos(raw_bearing))

            if true_range > SENSOR_RANGE_MAX or abs(true_bearing) > SENSOR_FOV / 2:
                continue

            obs = LandmarkObservation()
            obs.id = int(lm_id)
            obs.range = float(true_range + self.rng.normal(0, self.range_noise_std))
            obs.bearing = float(
                true_bearing + self.rng.normal(0, self.bearing_noise_std)
            )
            arr.observations.append(obs)

        self.obs_pub.publish(arr)

    def _publish_landmark_markers(self):
        markers = MarkerArray()
        for i, (lm_id, pos, color, name) in enumerate(LANDMARK_CONFIGS):
            m = Marker()
            m.header.frame_id = "world"
            m.ns = "ground_truth_landmarks"
            m.id = lm_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = pos
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color.r, m.color.g, m.color.b, m.color.a = color
            markers.markers.append(m)
        self.gt_landmarks_pub.publish(markers)

    def destroy_node(self):
        p.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LandmarkSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
