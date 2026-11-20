"""ROS message <-> planning_lab core conversions (kept out of core so core stays ROS-free and testable)."""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import OccupancyGrid

from sensor_motion_lab.core import GridMap
from sensor_motion_lab.core.geometry import FloatArray

from ..core.grid import PlanningGrid


class PlanningConv:
    OCCUPIED = 50  # OccupancyGrid value at/above which a cell is an obstacle (unknown = -1 counts as free)

    @classmethod
    def planning_grid(cls, msg: OccupancyGrid) -> PlanningGrid:
        info = msg.info
        occ = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width) >= cls.OCCUPIED
        return PlanningGrid(GridMap(occ, info.resolution, (info.origin.position.x, info.origin.position.y)))

    @staticmethod
    def points(msg: PoseArray) -> FloatArray:
        return np.array([[p.position.x, p.position.y] for p in msg.poses], dtype=np.float64).reshape(-1, 2)
