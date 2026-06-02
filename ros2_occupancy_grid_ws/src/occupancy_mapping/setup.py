# ros2_occupancy_grid_ws/src/occupancy_mapping/setup.py

from setuptools import find_packages, setup

package_name = "occupancy_mapping"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            "share/" + package_name + "/launch",
            ["launch/mapping.launch.py"],
        ),
    ],
    install_requires=[
        "setuptools",
    ],
    zip_safe=True,
    maintainer="Ashrith Sagar",
    maintainer_email="ashrithy@wilp.bits-pilani.ac.in",
    description="Occupancy grid mapping demo",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "mapping_node = occupancy_mapping.mapping_node:main",
        ],
    },
)
