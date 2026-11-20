import os
from glob import glob

from setuptools import find_packages, setup

package_name = "planning_lab"
NODES = {
    "planner_node": "planner_node:PlannerNode",
    "path_follower_node": "path_follower_node:PathFollowerNode",
    "dynamic_obstacles_node": "dynamic_obstacles_node:DynamicObstaclesNode",
    "navigation_evaluator_node": "navigation_evaluator_node:NavigationEvaluatorNode",
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
    install_requires=["setuptools", "numpy", "scipy", "matplotlib"],
    zip_safe=True,
    maintainer="Ashrith Sagar",
    maintainer_email="ashrithy@wilp.bits-pilani.ac.in",
    description="Webinar 4: A*, D* Lite and a genetic-algorithm waypoint planner replanning around moving "
                "people, benchmarked against the static-map optimum. Overlay on the W1 sensor_motion_lab workspace.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Each node class exposes a `main` classmethod (no free-floating functions).
            *(f"{exe} = {package_name}.ros.{mod_cls}.main" for exe, mod_cls in NODES.items()),
            f"planning_benchmark = {package_name}.benchmark:BenchmarkApp.main",
        ],
    },
)
