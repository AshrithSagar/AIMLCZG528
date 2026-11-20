"""
visual_slam_node — RGB-D visual SLAM as a ROS 2 node.

  sub  /camera/color/image_raw, /camera/depth/image_rect_raw, /camera/color/camera_info
       (time-synchronised with message_filters.ApproximateTimeSynchronizer)
  pub  TF  map -> camera_link                         (the SLAM estimate)
       /vslam/odometry         nav_msgs/Odometry      live pose
       /vslam/path             nav_msgs/Path          live (drifting) trajectory
       /vslam/optimized_path   nav_msgs/Path          after loop closure (pose-graph optimised)
       /vslam/keyframes        geometry_msgs/PoseArray
       /vslam/map/points       sensor_msgs/PointCloud2  sparse landmarks
       /vslam/map/dense        sensor_msgs/PointCloud2  coloured keyframe clouds (xyz + rgb)
       /vslam/loop_closures    visualization_msgs/MarkerArray
       /vslam/debug/features   sensor_msgs/Image      keypoints drawn on the image
       /diagnostics            per-stage timing, fps, dropped frames, tracking state
  srv  ~/save_trajectory (std_srvs/Trigger)  -> TUM files for `evo`
       ~/reset           (std_srvs/Trigger)

  param feature_type      := orb | sift | akaze | superpoint     (learned features: the GPU question)
        matcher           := bruteforce | lightglue               (learned matching)
        place_recognition := proximity | thumbnail | dinov2       (learned loop detection)
        device            := cpu | cuda                           (for every learned component)
  Depth may come from the sensor or from depth_estimator_node (learned monocular depth): same topic.
"""

from __future__ import annotations

import time
from pathlib import Path as FsPath

import cv2
import message_filters
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from ..core.camera import PinholeCamera, RgbdFrame
from ..core.pipeline import FrameResult, SlamConfig, VisualSlamPipeline
from ..core.tracking import OdometryConfig
from .common import Frames, MapConvention, QoS, RosConv, VslamNodeBase


class VisualSlamNode(VslamNodeBase):
    def __init__(self) -> None:
        super().__init__("visual_slam_node")
        self.config = SlamConfig(
            feature_type=self.param("feature_type", "orb"),
            matcher=self.param("matcher", "bruteforce"),
            place_recognition=self.param("place_recognition", "proximity"),
            max_features=self.param("max_features", 1000),
            device=self.param("device", "cpu"),
            loop_closure=self.param("loop_closure", True),
            odometry=OdometryConfig(max_depth=self.param("max_depth", 6.0)),
        )
        self.publish_tf = self.param("publish_tf", True)
        self.base_frame = self.param("base_frame", Frames.CAMERA)
        self.map_period = self.param("map_publish_period", 2.0)
        self.output_dir = FsPath(self.param("output_dir", str(FsPath.home() / "vslam_results"))).expanduser()

        self.bridge = CvBridge()
        self.pipeline: VisualSlamPipeline | None = None
        self.live_poses: list = []
        self.live_stamps: list = []
        self.received = 0
        self.processed = 0
        self.timings: dict[str, list[float]] = {}
        self.t_first: float | None = None
        self.last_map_pub = 0.0

        subs = [message_filters.Subscriber(self, Image, "/camera/color/image_raw", qos_profile=QoS.images()),
                message_filters.Subscriber(self, Image, "/camera/depth/image_rect_raw", qos_profile=QoS.images()),
                message_filters.Subscriber(self, CameraInfo, "/camera/color/camera_info", qos_profile=QoS.images())]
        self.sync = message_filters.ApproximateTimeSynchronizer(subs, queue_size=10, slop=self.param("sync_slop", 0.05))
        self.sync.registerCallback(self._on_rgbd)
        # count *every* colour frame, so we can report how many the SLAM had to drop
        self.create_subscription(Image, "/camera/color/image_raw", self._count, QoS.images())

        self.tf = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, "/vslam/odometry", 10)
        self.path_pub = self.create_publisher(Path, "/vslam/path", 10)
        self.opt_path_pub = self.create_publisher(Path, "/vslam/optimized_path", QoS.latched())
        self.kf_pub = self.create_publisher(PoseArray, "/vslam/keyframes", 10)
        self.points_pub = self.create_publisher(PointCloud2, "/vslam/map/points", QoS.latched())
        self.dense_pub = self.create_publisher(PointCloud2, "/vslam/map/dense", QoS.latched())
        self.loop_pub = self.create_publisher(MarkerArray, "/vslam/loop_closures", QoS.latched())
        self.debug_pub = self.create_publisher(Image, "/vslam/debug/features", QoS.images())
        self.create_service(Trigger, "~/save_trajectory", self._on_save)
        self.create_service(Trigger, "~/reset", self._on_reset)
        self.create_timer(self.param("report_period", 5.0), self._report)
        self.get_logger().info(f"Visual SLAM: {self.config.label} (features+matcher@place_recognition), "
                               f"device={self.config.device}, loop_closure={self.config.loop_closure}; "
                               "waiting for RGB-D frames ...")

    # ----- input ---------------------------------------------------------- #
    def _count(self, _msg: Image) -> None:
        self.received += 1

    def _on_rgbd(self, color: Image, depth: Image, info: CameraInfo) -> None:
        if self.pipeline is None:
            cam = PinholeCamera.from_k(info.k, info.width, info.height)
            self.pipeline = VisualSlamPipeline(cam, self.config)  # loads any learned model here (first frame)
            self.get_logger().info(f"Camera from CameraInfo: {cam}; models on {self.pipeline.extractor.device}")
        frame = RgbdFrame(RosConv.stamp_to_sec(color.header.stamp),
                          self.bridge.imgmsg_to_cv2(color, desired_encoding="rgb8"),
                          self._depth_metres(depth))
        t0 = time.perf_counter()
        result = self.pipeline.process(frame)
        result.timings_ms["callback_ms"] = 1000 * (time.perf_counter() - t0)
        self.processed += 1
        self.t_first = self.t_first or time.monotonic()
        for k, v in result.timings_ms.items():
            self.timings.setdefault(k, []).append(v)
        self._publish(result, color)

    def _depth_metres(self, msg: Image) -> np.ndarray:
        """REP-118: 32FC1 is metres, 16UC1 is millimetres."""
        d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if msg.encoding == "16UC1":
            return d.astype(np.float32) / 1000.0
        return np.nan_to_num(d.astype(np.float32), nan=0.0, posinf=0.0)

    # ----- output --------------------------------------------------------- #
    def _publish(self, r: FrameResult, color: Image) -> None:
        stamp = color.header.stamp
        T_map_body = MapConvention.pose(r.pose)  # the estimate is of the optical frame
        if self.publish_tf:
            self.tf.sendTransform(RosConv.transform(T_map_body, Frames.MAP, self.base_frame, stamp))
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, Frames.MAP, self.base_frame
        odom.pose.pose = RosConv.pose(T_map_body)
        self.odom_pub.publish(odom)

        self.live_poses.append(T_map_body)
        self.live_stamps.append(stamp)
        self.path_pub.publish(RosConv.path(self.live_poses[::2], self.live_stamps[::2], Frames.MAP))
        self.debug_pub.publish(self._debug_image(color, r))

        now = time.monotonic()
        if r.loop_closed or (r.new_keyframe and now - self.last_map_pub > self.map_period):
            self._publish_map(stamp)
            self.last_map_pub = now
        if r.loop_closed:
            self.get_logger().info(f"Loop closed! ({len(self.pipeline.map.loop_edges)} total), pose graph optimised")

    def _publish_map(self, stamp) -> None:  # noqa: ANN001
        assert self.pipeline is not None
        kmap = self.pipeline.map
        self.points_pub.publish(RosConv.point_cloud(MapConvention.points(kmap.sparse_points()), Frames.MAP, stamp))
        pts, cols = kmap.dense_points(self.pipeline.camera, stride=self.param("dense_stride", 8))
        self.dense_pub.publish(RosConv.point_cloud(MapConvention.points(pts), Frames.MAP, stamp, cols))
        kfs = PoseArray()
        kfs.header.frame_id, kfs.header.stamp = Frames.MAP, stamp
        kfs.poses = [RosConv.pose(MapConvention.pose(k.pose)) for k in kmap.keyframes]
        self.kf_pub.publish(kfs)
        traj = self.pipeline.trajectory()
        self.opt_path_pub.publish(RosConv.path([MapConvention.pose(p) for p in traj.poses[::2]],
                                               self.live_stamps[: len(traj.poses)][::2], Frames.MAP))
        self.loop_pub.publish(self._loop_markers(stamp))

    def _loop_markers(self, stamp) -> MarkerArray:  # noqa: ANN001
        assert self.pipeline is not None
        kmap = self.pipeline.map
        m = Marker()
        m.header.frame_id, m.header.stamp = Frames.MAP, stamp
        m.ns, m.id, m.type, m.action = "loops", 0, Marker.LINE_LIST, Marker.ADD
        m.scale.x = 0.03
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 1.0, 1.0
        for e in kmap.loop_edges:
            for kid in (e.i, e.j):
                p = RosConv.pose(MapConvention.pose(kmap.by_id(kid).pose)).position
                m.points.append(p)
        return MarkerArray(markers=[m])

    def _debug_image(self, color: Image, r: FrameResult) -> Image:
        assert self.pipeline is not None
        img = cv2.cvtColor(self.bridge.imgmsg_to_cv2(color, desired_encoding="rgb8"), cv2.COLOR_RGB2BGR)
        feats = self.pipeline.last_features
        colour = (0, 0, 255) if r.new_keyframe else (0, 255, 0)  # red dots = this frame became a keyframe
        for u, v in (feats.keypoints if feats is not None else []):
            cv2.circle(img, (int(u), int(v)), 2, colour, -1)
        txt = f"{self.config.label} {r.state.value} inl {r.inliers}/{r.matches} {r.timings_ms['total_ms']:.0f} ms"
        cv2.putText(img, txt, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        msg.header = color.header
        return msg

    # ----- metrics / services ---------------------------------------------- #
    def _report(self) -> None:
        if not self.processed or self.pipeline is None:
            return
        elapsed = max(time.monotonic() - (self.t_first or time.monotonic()), 1e-6)
        means = {k: float(np.mean(v[-100:])) for k, v in self.timings.items()}
        self.diagnostics.publish(
            {"config": self.config.label, "device": self.pipeline.extractor.device,
             "processed_fps": self.processed / elapsed, "input_fps": self.received / elapsed,
             "dropped_pct": 100.0 * (1 - self.processed / max(self.received, 1)),
             **{k: v for k, v in means.items()},
             "keyframes": len(self.pipeline.map.keyframes), "loops": len(self.pipeline.map.loop_edges)},
            message="processed_fps < input_fps means this machine cannot keep up (frames are dropped)",
        )

    def _on_save(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        if self.pipeline is None:
            resp.success, resp.message = False, "no frames processed yet"
            return resp
        path = self.pipeline.trajectory().write_tum(self.output_dir / f"vslam_{self.config.label}.tum")
        resp.success, resp.message = True, f"wrote {path} (optical-frame poses, TUM format)"
        return resp

    def _on_reset(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        self.pipeline, self.live_poses, self.live_stamps, self.timings = None, [], [], {}
        self.processed = self.received = 0
        self.t_first = None
        resp.success, resp.message = True, "SLAM reset; will re-initialise on the next frame"
        return resp

    def on_shutdown(self) -> None:
        if self.pipeline is not None and self.processed:
            try:
                self.pipeline.trajectory().write_tum(self.output_dir / f"vslam_{self.config.label}.tum")
            except OSError as exc:
                print(f"visual_slam_node: could not save trajectory: {exc}")
