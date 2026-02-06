from enum import Enum

import networkx as nx
from loguru import logger

from epog.envs.graph import SceneGraph
from pog.graph.node import ContainmentState
from pog.planning.action import Action, ActionType


class MotionErrorType(Enum):
    CollisionError = 0
    AccessError = 1
    StabilityError = 2
    BlockError = 3


class MotionError:
    def __init__(
        self,
        reason,
        error_type,
        failure_action,
        involved_nodes_id,
        observation_involved=None,
    ) -> None:
        self.reason: str = reason
        self.error_type: MotionErrorType = error_type  # used for the expert action
        self.failure_action: Action = failure_action
        self.parking_place: int = 0
        self.involved_nodes_id = involved_nodes_id
        self.observation_involved = observation_involved

    def __repr__(self) -> str:
        return f"""
            MotionError: {self.reason}, Error Type: {self.error_type},
            Failure Action: {self.failure_action}, Parking Place: {self.parking_place}
            observation_involved: {self.observation_involved}
        """


class FakeMotionPlanner:
    def __init__(self, goal_graph: SceneGraph) -> None:
        self.goal_graph = goal_graph

    def get_observation_involved(self, current: SceneGraph, node_involved: list[int]):
        nodes = []
        edges = []
        extend_nodes = []
        for edge in current.edge_dict.values():
            if edge.parent_id in node_involved or edge.child_id in node_involved:
                edges.append(edge.get_llm_info())
                extend_nodes.append(edge.parent_id)
                extend_nodes.append(edge.child_id)

        for node_id in set(extend_nodes):
            nodes.append(current.node_dict[node_id].get_llm_info())

        return {"nodes": nodes, "edges": edges}

    def update_graph(self, current: SceneGraph, action: Action):
        if action.action_type == ActionType.Pick:
            parent_id, child_id = action.del_edge
            child_node = current.node_dict[child_id]
            if child_node.is_virtual():  # remove exception node
                current.remove_virtual_nodes(child_id)
            else:
                current.removeNode(child_id)
        elif action.action_type == ActionType.Place:
            if self.is_contain_virtual_node(action):
                return False
            parent_id, child_id = action.add_edge
            if child_id in set(current.robot.nodes) and parent_id in set(
                current.robot.nodes
            ):
                logger.info(
                    f"Skip: {child_id} not in current.node_dict and {parent_id} not in current.node_dict it is in robot hand"
                )
                return True
            if (
                child_id in current.edge_dict
                and current.edge_dict[child_id].parent_id == parent_id
            ):
                logger.info(f"Skip: {child_id} has been placed in {parent_id}")
                return True
            current.addNode(parent_id, edge=self.goal_graph.edge_dict[child_id])
        elif action.action_type == ActionType.Open:
            current.node_dict[action.del_edge[1]].state = ContainmentState.Opened
        elif action.action_type == ActionType.Close:
            current.node_dict[action.del_edge[1]].state = ContainmentState.Closed
        return False

    def is_contain_virtual_node(self, action: Action):
        if action.action_type == ActionType.Pick:
            parent_id, child_id = action.del_edge
        elif action.action_type == ActionType.Place:
            parent_id, child_id = action.add_edge
        else:
            return False
        parent_node = self.goal_graph.node_dict[parent_id]
        child_node = self.goal_graph.node_dict[child_id]
        return parent_node.is_virtual() or child_node.is_virtual()

    def check_accessilbilty(self, graph: SceneGraph, action: Action):
        if action.action_type == ActionType.Pick:
            path_del = nx.shortest_path(
                graph.graph, source=graph.root, target=action.del_edge[1]
            )
            colsed_nodes = []
            for idx in range(len(path_del) - 1):
                if (
                    graph.node_dict[path_del[idx]].state == ContainmentState.Closed
                    and graph.edge_dict[path_del[idx + 1]].containment
                ):
                    colsed_nodes.append(path_del[idx])

            reason = f"object {action.del_edge[1]} is not accessible, because {colsed_nodes} is closed"
            error_type = MotionErrorType.AccessError
            return len(colsed_nodes) > 0, MotionError(
                reason, error_type, action, colsed_nodes, []
            )

        elif action.action_type == ActionType.Place:
            current = graph.copy()
            skip = self.update_graph(current, action)
            if skip:
                return False, None
            path_del = nx.shortest_path(
                current.graph, source=graph.root, target=action.add_edge[1]
            )
            colsed_nodes = []
            for idx in range(len(path_del) - 1):
                if (
                    current.node_dict[path_del[idx]].state == ContainmentState.Closed
                    and current.edge_dict[path_del[idx + 1]].containment
                ):
                    colsed_nodes.append(path_del[idx])

            reason = f"object {action.add_edge[1]} is not accessible, because {colsed_nodes} is closed"
            error_type = MotionErrorType.AccessError
            return len(colsed_nodes) > 0, MotionError(
                reason, error_type, action, colsed_nodes, []
            )
        return False, None

    def check_collision(self, graph: SceneGraph, action: Action):
        if action.action_type == ActionType.Place:
            parent_id, child_id = action.add_edge
            parent_node = graph.node_dict[parent_id]
            collision_nodes_id = parent_node.motion_info.collision_nodes_id
            if len(collision_nodes_id) > 0:
                reason = f"object {child_id} is in collision with {collision_nodes_id} on {parent_id}"
                relation = [
                    f"{node_id} on {parent_id}" for node_id in collision_nodes_id
                ]
                error_type = MotionErrorType.CollisionError
                return True, MotionError(
                    reason, error_type, action, collision_nodes_id, relation
                )
        return False, None

    def check_stability(self, graph: SceneGraph, action: Action):
        if action.action_type == ActionType.Pick:
            _, child_id = action.del_edge
            child_node = graph.node_dict[child_id]
            unstale_nodes_id = child_node.motion_info.unstable_nodes_id
            if len(unstale_nodes_id) > 0:
                reason = f"object {child_id} is not stable, because {unstale_nodes_id} is on {child_id}"
                error_type = MotionErrorType.StabilityError
                relation = [f"{node_id} on {child_id}" for node_id in unstale_nodes_id]
                return True, MotionError(
                    reason, error_type, action, unstale_nodes_id, relation
                )
        return False, None

    def check_block(self, graph: SceneGraph, action: Action):
        if action.action_type == ActionType.Pick:
            parent_id, child_id = action.del_edge
            child_node = graph.node_dict[child_id]
            block_nodes_id = child_node.motion_info.block_nodes_id
            if len(block_nodes_id) > 0:
                reason = f"object {child_id} is blocked by {block_nodes_id}"
                error_type = MotionErrorType.BlockError
                relation = [f"{node_id} on {parent_id}" for node_id in block_nodes_id]
                return True, MotionError(
                    reason, error_type, action, block_nodes_id, relation
                )
        return False, None

    def simulate_step(self, graph: SceneGraph, action: Action):
        if not self.is_contain_virtual_node(
            action
        ):  # virtual node operation will not rasie error
            # check accessilbilty
            unaccessible, messsage = self.check_accessilbilty(graph, action)
            if unaccessible:
                return unaccessible, messsage
            # check collision
            collision, messsage = self.check_collision(graph, action)
            if collision:
                return collision, messsage
            # check stability
            stability, messsage = self.check_stability(graph, action)
            if stability:
                return stability, messsage
            # check block
            block, messsage = self.check_block(graph, action)
            if block:
                return block, messsage
        # update graph
        self.update_graph(graph, action)
        return False, None


if __name__ == "__main__":
    pass
