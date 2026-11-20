import os
from glob import glob

from setuptools import find_packages, setup

package_name = "localization_lab"
NODES = {
    "localizer_node": "localizer_node:LocalizerNode",
    "pose_regressor_node": "pose_regressor_node:PoseRegressorNode",
    "localization_evaluator_node": "localization_evaluator_node:LocalizationEvaluatorNode",
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
    install_requires=["setuptools", "numpy", "scipy", "matplotlib"],
    zip_safe=True,
    maintainer="Ashrith Sagar",
    maintainer_email="ashrithy@wilp.bits-pilani.ac.in",
    description="Webinar 2: grid localization, (augmented) MCL and a learned scan->pose regressor, "
                "benchmarked against Nav2 AMCL. Overlay on the W1 sensor_motion_lab workspace.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Each node class exposes a `main` classmethod (no free-floating functions).
            *(f"{exe} = {package_name}.ros.{mod_cls}.main" for exe, mod_cls in NODES.items()),
            f"localization_benchmark = {package_name}.benchmark:BenchmarkApp.main",
            f"train_pose_regressor = {package_name}.benchmark:TrainApp.main",
        ],
    },
)
