"""Make the W1 underlay importable when tests run from a source checkout without `source install/setup.bash`."""

import sys
from pathlib import Path

try:
    import sensor_motion_lab  # noqa: F401
except ImportError:
    for parent in Path(__file__).resolve().parents:
        cand = parent / "ros2_sensor_motion_ws" / "src" / "sensor_motion_lab"
        if cand.is_dir():
            sys.path.insert(0, str(cand))
            break
