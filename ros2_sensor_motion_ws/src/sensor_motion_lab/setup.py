import os
from glob import glob

from setuptools import find_packages, setup

package_name = "sensor_motion_lab"
NODES = {
    "kinematic_sim_node": "kinematic_sim_node:KinematicSimNode",
    "world_map_node": "world_map_node:WorldMapNode",
    "umbmark_driver_node": "umbmark_driver_node:UMBmarkDriverNode",
    "motion_model_node": "motion_model_node:MotionModelNode",
    "sensor_model_node": "sensor_model_node:SensorModelNode",
    "trajectory_evaluator_node": "trajectory_evaluator_node:TrajectoryEvaluatorNode",
    "obstacle_avoidance_node": "obstacle_avoidance_node:ObstacleAvoidanceNode",
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
    ],
    install_requires=["setuptools", "numpy", "pyyaml", "matplotlib"],
    zip_safe=True,
    maintainer="Ashrith Sagar",
    maintainer_email="ashrithy@wilp.bits-pilani.ac.in",
    description="Sensor & motion modelling lab: beam / likelihood-field models, sampling motion models, UMBmark, ATE/RPE.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Each node class exposes a `main` classmethod (no free-floating functions).
            *(f"{exe} = {package_name}.ros.{mod_cls}.main" for exe, mod_cls in NODES.items()),
            f"offline_benchmark = {package_name}.offline_benchmark:main",
        ],
    },
)
