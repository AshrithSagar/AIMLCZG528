"""
prm_planner_node
==================
Probabilistic Roadmap (PRM). Samples N random free points once, connects
each to its k-nearest neighbours when the connecting segment is
collision-free, then runs Dijkstra over the resulting roadmap graph
(start/goal are wired in as extra nodes). Good contrast against RRT:
the roadmap is built once and can be reused / queried multiple times
(here we rebuild per query for simplicity, which is also easy to explain
live: "in a real system you'd cache this roadmap").

Live-tunable parameters:
  num_samples   (default 300)
  k_neighbors   (default 8)
  connect_radius (default 2.0 m) - max distance to attempt a connection
"""

import heapq
import random

import rclpy

from .planner_utils import PlannerNodeBase


class PRMPlannerNode(PlannerNodeBase):
    def __init__(self):
        super().__init__("prm_planner_node", marker_color=(0.0, 0.6, 0.3))
        self.declare_parameter("num_samples", 300)
        self.declare_parameter("k_neighbors", 8)
        self.declare_parameter("connect_radius", 2.0)

    def plan(self, start_xy, goal_xy):
        gm = self.grid_map
        n_samples = int(self.get_parameter("num_samples").value)
        k = int(self.get_parameter("k_neighbors").value)
        radius = self.get_parameter("connect_radius").value

        min_x, min_y = gm.origin_x, gm.origin_y
        max_x = gm.origin_x + gm.width * gm.resolution
        max_y = gm.origin_y + gm.height * gm.resolution

        # 1) sample free points, plus start (idx 0) and goal (idx 1)
        points = [start_xy, goal_xy]
        attempts = 0
        while len(points) < n_samples + 2 and attempts < n_samples * 20:
            attempts += 1
            p = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if gm.is_free_world(*p):
                points.append(p)

        # 2) connect k-nearest neighbours (collision-checked)
        adj = {i: [] for i in range(len(points))}
        edges_for_viz = []
        for i, p in enumerate(points):
            dists = sorted(
                (
                    ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2, j)
                    for j, q in enumerate(points)
                    if j != i
                )
            )
            connected = 0
            for d2, j in dists:
                if connected >= k:
                    break
                if d2**0.5 > radius:
                    continue
                if gm.line_free(*p, *points[j]):
                    adj[i].append((j, d2**0.5))
                    connected += 1
                    if i < j:
                        edges_for_viz.append((p, points[j]))

        roadmap_marker = self.make_line_list_marker(
            "prm_roadmap", edges_for_viz, rgba=(0.0, 0.6, 0.3, 0.35), width=0.008
        )

        # 3) Dijkstra from start(0) to goal(1)
        dist = {0: 0.0}
        prev = {}
        visited = set()
        heap = [(0.0, 0)]
        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == 1:
                break
            for v, w in adj[u]:
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if 1 not in dist:
            self.get_logger().warn(
                f"PRM: {len(points)} nodes sampled, roadmap disconnected "
                "-- try raising num_samples / connect_radius."
            )
            return None, [roadmap_marker]

        path_idx = [1]
        cur = 1
        while cur != 0:
            cur = prev[cur]
            path_idx.append(cur)
        path_idx.reverse()
        path = [points[i] for i in path_idx]

        self.get_logger().info(
            f"PRM: {len(points)} samples, {sum(len(v) for v in adj.values()) // 2} "
            f"edges, path uses {len(path_idx)} nodes."
        )
        return path, [roadmap_marker]


def main(args=None):
    rclpy.init(args=args)
    node = PRMPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
