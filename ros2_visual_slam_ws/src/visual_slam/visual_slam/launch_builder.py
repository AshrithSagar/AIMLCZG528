"""
Launch composition for every input source (DRY: one class, one launch file).

    source := synthetic   rgbd_player_node renders the CPU room          (no GPU, no download)
    source := tum         rgbd_player_node plays a TUM RGB-D folder       (real benchmark data)
    source := gazebo      gz-sim renders the same room with an RGB-D camera (OpenGL: the GPU test)
    source := bag         nothing — you run `ros2 bag play <bag>` yourself

Deep-learning switches (each one is a parameter of an existing node, or one extra node):

    feature_type:=superpoint        learned keypoints + descriptors
    matcher:=lightglue              learned matching (needs feature_type:=superpoint)
    place_recognition:=dinov2       learned loop detection (global image descriptor)
    depth:=learned                  depth_estimator_node predicts depth from colour; the player's
                                    sensor depth moves to /camera/depth_sensor/... (for scoring only)
    dl_trajectory:=<file.tum>       overlay an end-to-end DL SLAM result (DPVO / DROID-SLAM) in RViz
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.action import Action
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from .core.gazebo import GazeboWorldWriter


class VslamLaunchBuilder:
    PKG = "visual_slam"
    SOURCES = ("synthetic", "tum", "gazebo", "bag")
    SENSOR_DEPTH_TOPIC = "/camera/depth_sensor/image_rect_raw"
    STATIC_OPTICAL_TF = ["--qx", "-0.5", "--qy", "0.5", "--qz", "-0.5", "--qw", "0.5",
                         "--frame-id", "camera_link", "--child-frame-id", "camera_color_optical_frame"]

    def __init__(self) -> None:
        self.share = Path(get_package_share_directory(self.PKG))

    def build(self) -> LaunchDescription:
        args = [
            DeclareLaunchArgument("source", default_value="synthetic", choices=list(self.SOURCES)),
            DeclareLaunchArgument("path", default_value="", description="TUM sequence folder (source:=tum)"),
            DeclareLaunchArgument("rate", default_value="0.0", description="player rate in Hz (0 = native)"),
            DeclareLaunchArgument("feature_type", default_value="orb", choices=["orb", "sift", "akaze", "superpoint"]),
            DeclareLaunchArgument("matcher", default_value="bruteforce", choices=["bruteforce", "lightglue"]),
            DeclareLaunchArgument("place_recognition", default_value="proximity",
                                  choices=["proximity", "thumbnail", "dinov2"]),
            DeclareLaunchArgument("depth", default_value="sensor", choices=["sensor", "learned"]),
            DeclareLaunchArgument("depth_model_id", default_value="", description="Hugging Face id (default: "
                                  "Depth-Anything-V2-Metric-Indoor-Small)"),
            DeclareLaunchArgument("device", default_value="cpu", choices=["cpu", "cuda"]),
            DeclareLaunchArgument("dl_trajectory", default_value="",
                                  description="TUM file from DPVO / DROID-SLAM to overlay (needs source:=tum)"),
            DeclareLaunchArgument("loop_closure", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("publish_tf", default_value="true",
                                  description="false when another SLAM (e.g. RTAB-Map) owns map -> camera_link"),
        ]
        return LaunchDescription(args + [OpaqueFunction(function=self._setup)])

    # ------------------------------------------------------------------ #
    def _arg(self, ctx: LaunchContext, name: str) -> str:
        return LaunchConfiguration(name).perform(ctx)

    def _setup(self, ctx: LaunchContext) -> list[Action]:
        source = self._arg(ctx, "source")
        sim_time = source in ("gazebo", "bag")
        common = {"use_sim_time": sim_time}
        learned_depth = self._arg(ctx, "depth") == "learned"
        label = self._arg(ctx, "feature_type") + ("" if self._arg(ctx, "matcher") == "bruteforce"
                                                  else "+" + self._arg(ctx, "matcher"))
        actions: list[Action] = [
            Node(package=self.PKG, executable="visual_slam_node", output="screen", parameters=[common | {
                "feature_type": self._arg(ctx, "feature_type"),
                "matcher": self._arg(ctx, "matcher"),
                "place_recognition": self._arg(ctx, "place_recognition"),
                "device": self._arg(ctx, "device"),
                "loop_closure": self._arg(ctx, "loop_closure").lower() == "true",
                "publish_tf": self._arg(ctx, "publish_tf").lower() == "true",
            }]),
            Node(package=self.PKG, executable="trajectory_evaluator_node", output="screen", parameters=[common | {
                "ground_truth_type": "odometry" if source == "gazebo" else "pose",
                "label": label + ("_learned_depth" if learned_depth else ""),
            }]),
            Node(package="rviz2", executable="rviz2", output="log", parameters=[common],
                 arguments=["-d", str(self.share / "rviz" / "vslam.rviz")],
                 condition=IfCondition(LaunchConfiguration("rviz"))),
        ]
        if source in ("synthetic", "tum"):
            rate = float(self._arg(ctx, "rate"))
            actions.append(Node(package=self.PKG, executable="rgbd_player_node", output="screen", parameters=[{
                "source": source, "path": self._arg(ctx, "path"),
                "rate": rate if rate > 0 else (10.0 if source == "synthetic" else 0.0),
                "depth_topic": self.SENSOR_DEPTH_TOPIC if learned_depth else "/camera/depth/image_rect_raw",
            }]))
        else:
            actions.append(Node(package="tf2_ros", executable="static_transform_publisher", output="log",
                                arguments=self.STATIC_OPTICAL_TF, parameters=[common]))
        if source == "gazebo":
            actions += self._gazebo()
        if learned_depth:
            actions.append(Node(package=self.PKG, executable="depth_estimator_node", output="screen", parameters=[
                common | {"model_id": self._arg(ctx, "depth_model_id"), "device": self._arg(ctx, "device")}]))
        if self._arg(ctx, "dl_trajectory"):
            actions.append(self._overlay(ctx, common))
        return actions

    def _overlay(self, ctx: LaunchContext, common: dict) -> Node:
        return Node(package=self.PKG, executable="trajectory_overlay_node", output="screen", parameters=[common | {
            "trajectory_file": str(Path(self._arg(ctx, "dl_trajectory")).expanduser()),
            "sequence_path": str(Path(self._arg(ctx, "path")).expanduser()), "label": "dl_slam", "sim3": True}])

    def _gazebo(self) -> list[Action]:
        world = GazeboWorldWriter().write(Path(tempfile.gettempdir()) / "vslam_gazebo")
        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("ros_gz_sim"),
                                                           "launch", "gz_sim.launch.py")),
                launch_arguments={"gz_args": f"-r {world}", "on_exit_shutdown": "true"}.items()),
            Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
                 parameters=[{"config_file": str(self.share / "config" / "gz_bridge.yaml"), "use_sim_time": True}]),
            Node(package=self.PKG, executable="rig_driver_node", output="screen", parameters=[{"use_sim_time": True}]),
        ]
