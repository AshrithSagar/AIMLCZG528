# ros2_occupancy_grid_ws/src/occupancy_mapping/setup.py

from setuptools import find_packages, setup

setup(
    name="occupancy_mapping",
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/occupancy_mapping"],
        ),
        (
            "share/occupancy_mapping",
            ["package.xml"],
        ),
        (
            "share/occupancy_mapping/launch",
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
    license="MIT",
    entry_points={
        "console_scripts": [
            "mapping_node = occupancy_mapping.mapping_node:main",
        ],
    },
)
