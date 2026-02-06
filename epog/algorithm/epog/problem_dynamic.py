from epog.envs.proc_env import ProcEnv, update_graph
from epog.envs.proc_scene import SceneGraph
from pog.planning.action import Action
from pog.planning.problem import Arc, PlanningOnGraphProblem
from pog.planning.searchNode import SearchNode


class PlanningOnDynamicGraphProblem(PlanningOnGraphProblem):
    def __init__(
        self,
        start: SceneGraph,
        goal: SceneGraph,
        env: ProcEnv,
        action_seq: list[Action],
        constraints: dict[Action, list[Action]],
        navigation_map: dict[int, dict[int, float]] = None,
    ) -> None:
        self.start_graph = start
        self.goal_graph = goal
        self.env = env
        self.root_search_node = SearchNode(
            action_seq=action_seq, constraints=constraints, current=start, goal=goal
        )
        self.navigation_map = navigation_map  # for the navigation cost

    def neighbors(self, node: SearchNode) -> list[Arc]:
        """Find neighbors of current node

        Args:
            node (Graph): current node

        Returns:
            a list of Arc: all possible neighbors
        """
        neighbors = []
        for neighbor in node.selectAction():
            (new_action_seq, constraints, new_action) = neighbor
            current = node.current.copy()
            moving_cost, _ = update_graph(
                current, self.goal_graph, new_action, self.env
            )
            constraints.delConstraint(new_action)
            to_node = SearchNode(
                new_action_seq,
                constraints,
                new_action,
                current,
                self.goal_graph,
                moving_cost,
            )
            neighbors.append(Arc(node, to_node))
        return neighbors
