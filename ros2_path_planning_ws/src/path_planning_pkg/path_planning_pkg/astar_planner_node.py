"""
astar_planner_node
===================
Classic grid-based A* search over the occupancy grid (8-connected,
octile heuristic). Optimal (given the grid resolution) but explores
a search-effort proportional to grid size -- good contrast against
the sampling-based planners.
"""

import heapq
import math

import rclpy

from .planner_utils import PlannerNodeBase

# 8-connected neighbourhood: (dx, dy, step_cost)
NEIGHBORS = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (-1, -1, math.sqrt(2)),
]


class AStarPlannerNode(PlannerNodeBase):
    def __init__(self):
        super().__init__("astar_planner_node", marker_color=(1.0, 0.5, 0.0))
        # No extra parameters needed for vanilla A*, but exposed for tuning
        self.declare_parameter("heuristic_weight", 1.0)

    def plan(self, start_xy, goal_xy):
        gm = self.grid_map
        w = self.get_parameter("heuristic_weight").value

        start = gm.world_to_grid(*start_xy)
        goal = gm.world_to_grid(*goal_xy)

        def h(a, b):
            dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
            return w * (max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy))  # octile

        open_heap = [(h(start, goal), 0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        closed = set()
        explored_edges = []  # for visualization (kept light: only expansions)

        while open_heap:
            _, g, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)

            if current == goal:
                return self._reconstruct(came_from, current, gm), [
                    self._explored_marker(explored_edges)
                ]

            for dx, dy, cost in NEIGHBORS:
                nxt = (current[0] + dx, current[1] + dy)
                if not gm.is_free_cell(*nxt):
                    continue
                tentative_g = g + cost
                if tentative_g < g_score.get(nxt, float("inf")):
                    g_score[nxt] = tentative_g
                    came_from[nxt] = current
                    heapq.heappush(
                        open_heap, (tentative_g + h(nxt, goal), tentative_g, nxt)
                    )
                    explored_edges.append((current, nxt))

        self.get_logger().warn(f"A*: explored {len(closed)} cells, no path exists.")
        return None, [self._explored_marker(explored_edges)]

    def _reconstruct(self, came_from, current, gm):
        path_grid = [current]
        while current in came_from:
            current = came_from[current]
            path_grid.append(current)
        path_grid.reverse()
        return [gm.grid_to_world(gx, gy) for gx, gy in path_grid]

    def _explored_marker(self, explored_edges):
        gm = self.grid_map
        segs = []
        # subsample for rendering performance during live demo
        for a, b in explored_edges[::3]:
            segs.append((gm.grid_to_world(*a), gm.grid_to_world(*b)))
        return self.make_line_list_marker(
            "astar_explored", segs, rgba=(1.0, 0.7, 0.0, 0.25), width=0.01
        )


def main(args=None):
    rclpy.init(args=args)
    node = AStarPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
