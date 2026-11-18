"""
world_map_node — publishes the environment as ROS maps.

  /map               nav_msgs/OccupancyGrid  (latched)   rasterised from world.yaml
  /likelihood_field  nav_msgs/OccupancyGrid  (latched)   100·exp(-d²/2σ²), the "map" of the LF model
  TF  map -> odom    identity (static)                   REP-105: a localiser would own this;
                                                         here the robot starts at the map origin.
"""

from __future__ import annotations

import numpy as np
from nav_msgs.msg import OccupancyGrid
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from ..core import GridRasterizer, Pose2D
from .common import LabNode, QoS, RosConv


class WorldMapNode(LabNode):
    def __init__(self) -> None:
        super().__init__("world_map_node")
        resolution = self.param("resolution", 0.05)
        sigma = self.param("sigma_hit", 0.2)
        max_dist = self.param("max_dist", 2.0)

        grid = GridRasterizer(resolution).rasterize(self.world)
        stamp = self.now_msg()

        self.map_pub = self.create_publisher(OccupancyGrid, "/map", QoS.latched())
        self.lf_pub = self.create_publisher(OccupancyGrid, "/likelihood_field", QoS.latched())
        self.map_pub.publish(RosConv.grid_map(grid, self.MAP_FRAME, stamp))

        lf = 100.0 * np.exp(-0.5 * (np.minimum(grid.distance_field, max_dist) / sigma) ** 2)
        self.lf_pub.publish(RosConv.occupancy_grid(np.round(lf), grid.resolution,
                                                   (grid.origin_x, grid.origin_y), self.MAP_FRAME, stamp))

        self.static_tf = StaticTransformBroadcaster(self)
        self.static_tf.sendTransform(RosConv.transform(Pose2D(), self.MAP_FRAME, self.ODOM_FRAME, stamp))
        self.get_logger().info(
            f"Published /map ({grid.width}x{grid.height} @ {resolution} m) and /likelihood_field "
            f"for world '{self.world.name}'."
        )
