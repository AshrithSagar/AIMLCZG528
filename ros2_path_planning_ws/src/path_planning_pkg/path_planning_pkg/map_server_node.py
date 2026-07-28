"""
map_server_node
================
Publishes a fixed demo occupancy grid on /map (latched / transient-local QoS
so RViz2 and planner nodes started later still receive it).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

from .planner_utils import make_demo_map


class MapServerNode(Node):
    def __init__(self):
        super().__init__("map_server_node")

        qos = QoSProfile(depth=1)
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = QoSReliabilityPolicy.RELIABLE

        self.pub = self.create_publisher(type(make_demo_map()), "/map", qos)

        self.map_msg = make_demo_map()
        self.map_msg.header.frame_id = "map"

        # Re-stamp and republish periodically so tools that don't honour
        # transient-local (or join late) still pick it up reliably.
        self.timer = self.create_timer(1.0, self.publish_map)
        self.get_logger().info(
            f"Publishing demo map: {self.map_msg.info.width}x"
            f"{self.map_msg.info.height} cells @ "
            f"{self.map_msg.info.resolution} m/cell on /map"
        )

    def publish_map(self):
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.map_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MapServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
