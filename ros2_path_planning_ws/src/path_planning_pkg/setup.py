import os
from glob import glob

from setuptools import find_packages, setup

package_name = "path_planning_pkg"

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
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Ashrith Sagar",
    maintainer_email="ashrithy@wilp.bits-pilani.ac.in",
    description="RRT / PRM / A* path planners on an occupancy grid.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "map_server_node = path_planning_pkg.map_server_node:main",
            "astar_planner_node = path_planning_pkg.astar_planner_node:main",
            "rrt_planner_node = path_planning_pkg.rrt_planner_node:main",
            "prm_planner_node = path_planning_pkg.prm_planner_node:main",
        ],
    },
)
