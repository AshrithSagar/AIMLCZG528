import os
from glob import glob

from setuptools import find_packages, setup

package_name = "visual_slam"
NODES = {
    "rgbd_player_node": "rgbd_player_node:RgbdPlayerNode",
    "visual_slam_node": "visual_slam_node:VisualSlamNode",
    "trajectory_evaluator_node": "trajectory_evaluator_node:TrajectoryEvaluatorNode",
    "rig_driver_node": "rig_driver_node:RigDriverNode",
    "depth_estimator_node": "depth_estimator_node:DepthEstimatorNode",
    "trajectory_overlay_node": "trajectory_overlay_node:TrajectoryOverlayNode",
}

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy", "opencv-python", "scipy", "matplotlib"],
    zip_safe=True,
    maintainer="Ashrith Sagar",
    maintainer_email="ashrithy@wilp.bits-pilani.ac.in",
    description="RGB-D visual SLAM in ROS 2 with optional deep learning: SuperPoint + LightGlue matching, "
                "DINOv2 loop detection, learned monocular depth; PnP odometry, pose-graph optimisation, "
                "TUM ATE/RPE evaluation.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Each node class exposes a `main` classmethod (no free-floating functions).
            *(f"{exe} = {package_name}.ros.{mod_cls}.main" for exe, mod_cls in NODES.items()),
            f"vslam_benchmark = {package_name}.benchmark:BenchmarkApp.main",
            f"export_synthetic = {package_name}.benchmark:ExportApp.main",
            f"vslam_score = {package_name}.apps:ScoreApp.main",
            f"dl_check = {package_name}.apps:DlCheckApp.main",
        ],
    },
)
