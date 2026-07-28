"""
rrt_planner_node
=================
Rapidly-exploring Random Tree (RRT). Grows a tree from the start by
sampling random free points and steering towards them, with goal
biasing. Publishes the full tree (LINE_LIST marker) plus the final
path for visually contrasting its "space-filling" exploration
against A*'s grid sweep and PRM's roadmap.

Live-tunable parameters (ros2 param set /rrt_planner_node <name> <value>):
  step_size    (default 0.3 m)  - branch length per extension
  goal_bias    (default 0.08)   - probability of sampling the goal directly
  max_iter     (default 5000)   - iteration budget
  goal_tol     (default 0.25 m) - success radius around goal
"""

import random

import rclpy

from .planner_utils import PlannerNodeBase


class RRTPlannerNode(PlannerNodeBase):
    def __init__(self):
        super().__init__("rrt_planner_node", marker_color=(0.6, 0.1, 0.8))
        self.declare_parameter("step_size", 0.3)
        self.declare_parameter("goal_bias", 0.08)
        self.declare_parameter("max_iter", 5000)
        self.declare_parameter("goal_tol", 0.25)

    def plan(self, start_xy, goal_xy):
        gm = self.grid_map
        step = self.get_parameter("step_size").value
        goal_bias = self.get_parameter("goal_bias").value
        max_iter = int(self.get_parameter("max_iter").value)
        goal_tol = self.get_parameter("goal_tol").value

        min_x = gm.origin_x
        min_y = gm.origin_y
        max_x = gm.origin_x + gm.width * gm.resolution
        max_y = gm.origin_y + gm.height * gm.resolution

        nodes = [start_xy]
        parent = {0: None}

        def nearest_idx(pt):
            return min(
                range(len(nodes)),
                key=lambda i: (nodes[i][0] - pt[0]) ** 2 + (nodes[i][1] - pt[1]) ** 2,
            )

        def steer(a, b, max_step):
            dx, dy = b[0] - a[0], b[1] - a[1]
            dist = (dx**2 + dy**2) ** 0.5
            if dist <= max_step:
                return b
            r = max_step / dist
            return (a[0] + dx * r, a[1] + dy * r)

        goal_idx = None
        for _ in range(max_iter):
            if random.random() < goal_bias:
                sample = goal_xy
            else:
                sample = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))

            i_near = nearest_idx(sample)
            new_pt = steer(nodes[i_near], sample, step)

            if not gm.is_free_world(*new_pt):
                continue
            if not gm.line_free(*nodes[i_near], *new_pt):
                continue

            nodes.append(new_pt)
            parent[len(nodes) - 1] = i_near

            if (
                (new_pt[0] - goal_xy[0]) ** 2 + (new_pt[1] - goal_xy[1]) ** 2
            ) ** 0.5 <= goal_tol:
                goal_idx = len(nodes) - 1
                break

        tree_marker = self._tree_marker(nodes, parent)

        if goal_idx is None:
            self.get_logger().warn(
                f"RRT: exhausted {max_iter} iterations ({len(nodes)} nodes), no path."
            )
            return None, [tree_marker]

        # reconstruct path
        path = []
        idx = goal_idx
        while idx is not None:
            path.append(nodes[idx])
            idx = parent[idx]
        path.reverse()
        path.append(goal_xy)

        self.get_logger().info(f"RRT: tree has {len(nodes)} nodes.")
        return path, [tree_marker]

    def _tree_marker(self, nodes, parent):
        segs = []
        for idx, p_idx in parent.items():
            if p_idx is None:
                continue
            segs.append((nodes[p_idx], nodes[idx]))
        return self.make_line_list_marker(
            "rrt_tree", segs, rgba=(0.6, 0.1, 0.8, 0.5), width=0.01
        )


def main(args=None):
    rclpy.init(args=args)
    node = RRTPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
