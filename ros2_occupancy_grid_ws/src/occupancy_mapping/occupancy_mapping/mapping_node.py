#!/usr/bin/env python3

# ros2_occupancy_grid_ws/src/occupancy_mapping/mapping_node.py

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

# Building specs: (cx, cy, half_x, half_y, height, rgba)
BUILDINGS = [
    (5.0, 5.0, 1.5, 1.5, 1.5, "0.80 0.40 0.40 1"),
    (-5.0, 5.0, 1.5, 1.0, 1.0, "0.40 0.60 0.80 1"),
    (5.0, -5.0, 1.0, 1.5, 2.0, "0.60 0.50 0.75 1"),
    (-5.0, -5.0, 2.0, 1.0, 1.2, "0.70 0.70 0.40 1"),
    (0.0, 7.5, 1.5, 1.0, 1.0, "0.50 0.70 0.50 1"),
    (0.0, -7.5, 1.5, 1.0, 0.8, "0.70 0.50 0.30 1"),
    (8.5, 0.0, 0.8, 2.0, 1.0, "0.40 0.50 0.65 1"),
    (-8.5, 0.0, 0.8, 1.5, 1.2, "0.60 0.40 0.55 1"),
]


def build_city_xml():
    """Generate the MuJoCo XML for the city scene."""
    geom_xml = "\n".join(
        f'        <geom name="b{i}" type="box" pos="{cx} {cy} {h / 2}" '
        f'size="{hx} {hy} {h / 2}" rgba="{rgba}"/>'
        for i, (cx, cy, hx, hy, h, rgba) in enumerate(BUILDINGS)
    )
    return f"""
<mujoco model="city">
    <option timestep="0.02"/>
    <visual>
        <headlight diffuse="0.7 0.7 0.7" ambient="0.35 0.35 0.35" specular="0 0 0"/>
        <global offwidth="640" offheight="480"/>
    </visual>
    <asset>
        <texture name="sky" type="skybox" builtin="gradient"
                 rgb1="0.5 0.7 0.95" rgb2="0.85 0.9 1.0"
                 width="256" height="256"/>
        <texture name="grid" type="2d" builtin="checker"
                 rgb1="0.35 0.55 0.35" rgb2="0.30 0.50 0.30"
                 width="512" height="512"/>
        <material name="grid_mat" texture="grid" texrepeat="10 10"/>
    </asset>
    <worldbody>
        <light pos="0 0 15" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <light pos="6 6 8"  dir="-1 -1 -1" diffuse="0.3 0.3 0.3"/>
        <camera name="chase"   pos="0 -14 9" xyaxes="1 0 0 0 0.6 0.8"/>
        <camera name="topdown" pos="0 0 24" euler="0 0 0" fovy="60"/>
        <geom name="floor" type="plane" size="15 15 0.1" material="grid_mat"/>
{geom_xml}
        <body name="car" pos="0 0 0.15">
            <joint name="car_x"   type="slide" axis="1 0 0" damping="0.5"/>
            <joint name="car_y"   type="slide" axis="0 1 0" damping="0.5"/>
            <joint name="car_yaw" type="hinge" axis="0 0 1" damping="0.5"/>
            <geom name="car_body" type="box" size="0.35 0.20 0.08"
                  rgba="0.10 0.30 0.95 1"/>
            <geom name="car_top"  type="box" pos="-0.05 0 0.10"
                  size="0.20 0.15 0.05" rgba="0.05 0.15 0.6 1"/>
            <geom name="car_arrow" type="box" pos="0.33 0 0"
                  size="0.07 0.05 0.03" rgba="1 1 0 1"/>
            <site name="lidar" pos="0 0 0.18" size="0.04" rgba="1 0 0 1"/>
        </body>
    </worldbody>
</mujoco>
"""


class BeamSensorModel:
    """LiDAR-like beam sensor backed by MuJoCo ray casting."""

    def __init__(
        self,
        n_beams=72,
        max_range=9.0,
        sigma_hit=0.05,
        p_short=0.01,
        p_max=0.02,
        p_rand=0.01,
        exclude_body=-1,
    ):
        self.n_beams = n_beams
        self.max_range = max_range
        self.sigma_hit = sigma_hit
        self.p_short, self.p_max, self.p_rand = p_short, p_max, p_rand
        self.exclude_body = exclude_body
        self.beam_angles = np.linspace(0.0, 2 * np.pi, n_beams, endpoint=False)

    def cast_rays(self, model, data, sensor_pos, sensor_yaw, add_noise=True):
        distances = np.full(self.n_beams, self.max_range)
        hits = np.zeros((self.n_beams, 2))
        origin = np.asarray(sensor_pos, dtype=np.float64)
        geomid = np.zeros(1, dtype=np.int32)
        for i, a in enumerate(self.beam_angles):
            theta = sensor_yaw + a
            direction = np.array([np.cos(theta), np.sin(theta), 0.0])
            d = mujoco.mj_ray(
                model, data, origin, direction, None, 1, self.exclude_body, geomid
            )
            if d < 0 or d > self.max_range:
                d = self.max_range
            if add_noise:
                d = self._noisify(d)
            distances[i] = d
            hits[i] = origin[:2] + d * direction[:2]
        return distances, hits

    def _noisify(self, z_true):
        u = np.random.rand()
        p_hit = 1.0 - self.p_short - self.p_max - self.p_rand
        if u < p_hit:
            return float(
                np.clip(
                    z_true + np.random.randn() * self.sigma_hit, 0.0, self.max_range
                )
            )
        if u < p_hit + self.p_short and z_true > 0:
            return float(np.random.uniform(0.0, z_true))
        if u < p_hit + self.p_short + self.p_max:
            return float(self.max_range)
        return float(np.random.uniform(0.0, self.max_range))


class VelocityMotionModel:
    """Probabilistic velocity motion model (Thrun ch. 5)."""

    def __init__(self, alpha=(0.03, 0.02, 0.06, 0.03, 0.0, 0.0)):
        self.alpha = alpha

    def sample(self, state, u, dt):
        """Return a NOISY next state."""
        v, w = u
        a1, a2, a3, a4, a5, a6 = self.alpha
        v_h = v + np.random.randn() * np.sqrt(a1 * v * v + a2 * w * w + 1e-9)
        w_h = w + np.random.randn() * np.sqrt(a3 * v * v + a4 * w * w + 1e-9)
        gamma = np.random.randn() * np.sqrt(a5 * v * v + a6 * w * w + 1e-9)
        return self.integrate(state, v_h, w_h, dt, gamma)

    @staticmethod
    def integrate(state, v, w, dt, gamma=0.0):
        """Deterministic integration (used for ground truth)."""
        x, y, theta = state
        if abs(w) < 1e-6:
            xn = x + v * np.cos(theta) * dt
            yn = y + v * np.sin(theta) * dt
            thn = theta + gamma * dt
        else:
            r = v / w
            xn = x - r * np.sin(theta) + r * np.sin(theta + w * dt)
            yn = y + r * np.cos(theta) - r * np.cos(theta + w * dt)
            thn = theta + w * dt + gamma * dt
        thn = (thn + np.pi) % (2 * np.pi) - np.pi
        return np.array([xn, yn, thn])


class SimpleLocalizer:
    """Open-loop dead-reckoning localizer."""

    def __init__(self, initial_pose, motion_model):
        self.pose = np.array(initial_pose, dtype=float)
        self.motion = motion_model
        self.trajectory = [self.pose.copy()]

    def update(self, control, dt):
        self.pose = self.motion.sample(self.pose, control, dt)
        self.trajectory.append(self.pose.copy())
        return self.pose


class OccupancyGridMap:
    """Log-odds occupancy grid (Moravec & Elfes, 1985)."""

    def __init__(
        self, size=30.0, resolution=0.15, l_occ=0.85, l_free=-0.4, l_min=-2.0, l_max=3.5
    ):
        self.size = size
        self.resolution = resolution
        self.n = int(size / resolution)
        self.origin_cell = self.n // 2
        self.l_occ, self.l_free = l_occ, l_free
        self.l_min, self.l_max = l_min, l_max
        self.log_odds = np.zeros((self.n, self.n), dtype=np.float32)

    def world_to_grid(self, x, y):
        return (
            int(np.floor(x / self.resolution)) + self.origin_cell,
            int(np.floor(y / self.resolution)) + self.origin_cell,
        )

    def in_bounds(self, gx, gy):
        return 0 <= gx < self.n and 0 <= gy < self.n

    def update(self, pose, distances, beam_angles, max_range):
        x, y, theta = pose
        gx0, gy0 = self.world_to_grid(x, y)
        for d, a in zip(distances, beam_angles):
            wa = theta + a
            ex = x + d * np.cos(wa)
            ey = y + d * np.sin(wa)
            gx1, gy1 = self.world_to_grid(ex, ey)
            cells = self._bresenham(gx0, gy0, gx1, gy1)
            for cx, cy in cells[:-1]:
                if self.in_bounds(cx, cy):
                    self.log_odds[cx, cy] += self.l_free
            if d < max_range - 1e-3 and self.in_bounds(gx1, gy1):
                self.log_odds[gx1, gy1] += self.l_occ
        np.clip(self.log_odds, self.l_min, self.l_max, out=self.log_odds)

    @staticmethod
    def _bresenham(x0, y0, x1, y1):
        cells = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        for _ in range(dx + dy + 1):
            cells.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return cells

    def probability(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))


class MappingNode(Node):
    def __init__(self):

        super().__init__("mapping_node")

        # --------------------------------------------------
        # QoS
        # --------------------------------------------------

        map_qos = QoSProfile(depth=1)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        # --------------------------------------------------
        # Publishers
        # --------------------------------------------------

        self.scan_pub = self.create_publisher(
            LaserScan,
            "/scan",
            10,
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            "/odom",
            10,
        )

        self.map_pub = self.create_publisher(
            OccupancyGrid,
            "/map",
            map_qos,
        )

        # --------------------------------------------------
        # TF
        # --------------------------------------------------

        self.tf_broadcaster = TransformBroadcaster(self)

        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.publish_static_tf()

        # --------------------------------------------------
        # MuJoCo
        # --------------------------------------------------

        self.model = mujoco.MjModel.from_xml_string(build_city_xml())

        self.data = mujoco.MjData(self.model)

        self.car_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "car",
        )

        self.lidar_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "lidar",
        )

        # --------------------------------------------------
        # Sensor
        # --------------------------------------------------

        self.sensor = BeamSensorModel(
            n_beams=72,
            max_range=9.0,
            exclude_body=self.car_body_id,
        )

        # --------------------------------------------------
        # Motion model
        # --------------------------------------------------

        self.motion = VelocityMotionModel()

        self.true_pose = np.array(
            [0.0, 0.0, 0.0],
            dtype=float,
        )

        self.localizer = SimpleLocalizer(
            self.true_pose.copy(),
            self.motion,
        )

        # --------------------------------------------------
        # Occupancy grid
        # --------------------------------------------------

        self.grid = OccupancyGridMap(
            size=30.0,
            resolution=0.15,
        )

        # --------------------------------------------------
        # Simulation timing
        # --------------------------------------------------

        self.dt = 0.12

        self.timer = self.create_timer(
            self.dt,
            self.step,
        )

        self.get_logger().info("MuJoCo occupancy mapping started")

    # ======================================================
    # Static TF
    # ======================================================

    def publish_static_tf(self):

        tf = TransformStamped()

        tf.header.stamp = self.get_clock().now().to_msg()

        tf.header.frame_id = "base_link"
        tf.child_frame_id = "laser"

        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.18

        tf.transform.rotation.w = 1.0

        self.static_tf_broadcaster.sendTransform(tf)

    # ======================================================
    # Main simulation step
    # ======================================================

    def step(self):

        # constant velocity circular path

        v = 1.0
        w = 0.35

        # ---------------------------------------------
        # ground truth
        # ---------------------------------------------

        self.true_pose = self.motion.integrate(
            self.true_pose,
            v,
            w,
            self.dt,
        )

        self.data.qpos[:3] = self.true_pose

        self.data.qvel[:] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        # ---------------------------------------------
        # lidar
        # ---------------------------------------------

        sensor_pos = self.data.site_xpos[self.lidar_id].copy()

        distances, _ = self.sensor.cast_rays(
            self.model,
            self.data,
            sensor_pos,
            self.true_pose[2],
        )

        # ---------------------------------------------
        # localization
        # ---------------------------------------------

        est_pose = self.localizer.update(
            [v, w],
            self.dt,
        )

        # ---------------------------------------------
        # mapping
        # ---------------------------------------------

        self.grid.update(
            est_pose,
            distances,
            self.sensor.beam_angles,
            self.sensor.max_range,
        )

        # ---------------------------------------------
        # ROS publications
        # ---------------------------------------------

        self.publish_scan(distances)

        self.publish_odom(est_pose)

        self.publish_tf(est_pose)

        self.publish_map()

    # ======================================================
    # Laser
    # ======================================================

    def publish_scan(self, distances):

        msg = LaserScan()

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.header.frame_id = "laser"

        msg.angle_min = 0.0

        msg.angle_max = 2.0 * np.pi

        msg.angle_increment = 2.0 * np.pi / self.sensor.n_beams

        msg.range_min = 0.05

        msg.range_max = self.sensor.max_range

        msg.ranges = distances.tolist()

        self.scan_pub.publish(msg)

    # ======================================================
    # Odom
    # ======================================================

    def publish_odom(self, pose):

        msg = Odometry()

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.header.frame_id = "map"

        msg.child_frame_id = "base_link"

        msg.pose.pose.position.x = float(pose[0])

        msg.pose.pose.position.y = float(pose[1])

        yaw = float(pose[2])

        msg.pose.pose.orientation.z = np.sin(yaw / 2.0)

        msg.pose.pose.orientation.w = np.cos(yaw / 2.0)

        msg.pose.covariance[0] = 0.05
        msg.pose.covariance[7] = 0.05
        msg.pose.covariance[35] = 0.02

        self.odom_pub.publish(msg)

    # ======================================================
    # Dynamic TF
    # ======================================================

    def publish_tf(self, pose):

        tf = TransformStamped()

        tf.header.stamp = self.get_clock().now().to_msg()

        tf.header.frame_id = "map"

        tf.child_frame_id = "base_link"

        tf.transform.translation.x = float(pose[0])

        tf.transform.translation.y = float(pose[1])

        tf.transform.translation.z = 0.0

        yaw = float(pose[2])

        tf.transform.rotation.z = np.sin(yaw / 2.0)

        tf.transform.rotation.w = np.cos(yaw / 2.0)

        self.tf_broadcaster.sendTransform(tf)

    # ======================================================
    # Occupancy Grid
    # ======================================================

    def publish_map(self):

        prob = self.grid.probability()

        grid = np.full(
            prob.shape,
            -1,
            dtype=np.int8,
        )

        grid[prob < 0.30] = 0

        grid[prob > 0.70] = 100

        msg = OccupancyGrid()

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.header.frame_id = "map"

        msg.info.resolution = self.grid.resolution

        msg.info.width = self.grid.n

        msg.info.height = self.grid.n

        msg.info.origin.position.x = -self.grid.size / 2.0

        msg.info.origin.position.y = -self.grid.size / 2.0

        msg.info.origin.orientation.w = 1.0

        # row-major flatten

        msg.data = grid.flatten().astype(np.int8).tolist()

        self.map_pub.publish(msg)


def main():

    rclpy.init()

    node = MappingNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
