import random

from loguru import logger

from epog.algorithm.epog.fake_simulator import FakeMotionPlanner
from epog.algorithm.epog.planner_dynamic import Planner
from epog.algorithm.epog.problem_dynamic import PlanningOnDynamicGraphProblem
from epog.envs.graph import SceneEdge, SceneGraph, SceneNode
from epog.envs.proc_env import Observation, ProcEnv, print_action, update_graph
from pog.planning.action import Action, ActionType
from pog.planning.ged import ged_seq
from pog.planning.planner import Searcher, test
from pog.planning.utils import path_to_action_sequence


class PlanningMetrics:
    def __init__(
        self,
        moving_cost: float = 0,
        expand_node: int = 0,
        action_seq_len: int = 0,
        success: bool = False,
    ) -> None:
        self.success = success
        self.moving_cost = moving_cost
        self.expand_node = expand_node
        self.action_seq_len = action_seq_len

    def __repr__(self) -> str:
        return f"Moving Cost: {self.moving_cost}, Expand Node: {self.expand_node}, Action Seq Len: {self.action_seq_len}"

    def __add__(self, other):
        return PlanningMetrics(
            self.moving_cost + other.moving_cost,
            self.expand_node + other.expand_node,
            self.action_seq_len + other.action_seq_len,
        )

    def __truediv__(self, other: int):
        return PlanningMetrics(
            self.moving_cost / other,
            self.expand_node / other,
            self.action_seq_len / other,
        )


class EPoG:
    def __init__(
        self,
        env: ProcEnv,
        global_agent: str = "expert",
        local_agent: str = "expert",
        path_opt: bool = False,
        logger=None,
    ) -> None:
        self.env = env
        self.belief_graph = env.get_init_graph()
        self.task_graph = env.task_graph
        self.path_opt = path_opt
        self.agent = Planner(
            global_agent=global_agent,
            local_agent=local_agent,
            state_graph=self.env.state_graph,
        )
        self.simulator = FakeMotionPlanner(goal_graph=self.task_graph)
        self.traj: list[Action] = []
        self.logger = logger

    def get_lost_nodes(self) -> list[int]:
        node_ids = []
        for node_id in self.task_graph.node_dict:
            if node_id not in self.belief_graph.node_dict:
                node_ids.append(node_id)
        for node_id in self.belief_graph.node_dict:
            assert (
                node_id in self.task_graph.node_dict
                or self.belief_graph.node_dict[node_id].is_virtual()
            )
        return node_ids

    def init_visited_map(self) -> tuple[dict, dict]:
        # add room nodes to the belif graph
        self.room_visited, self.receptacle_visited = {}, {}
        for room_node in self.belief_graph.room_nodes:
            self.room_visited[room_node.id] = False
            self.receptacle_visited[room_node.id] = {}
            for node in self.belief_graph.receptacle_nodes[room_node.id]:
                self.receptacle_visited[room_node.id][node.id] = False

    def check_room_visited(self, room_id: int) -> bool:
        # room visited = room visited and receptacle visited
        for receptacle_id in self.receptacle_visited[room_id]:
            if not self.receptacle_visited[room_id][receptacle_id]:
                return False
        return True

    def get_unvisited_rooms_type(self):
        unvisited_rooms_type = set()
        for room_id, _visited in self.room_visited.items():
            if not self.check_room_visited(room_id):
                room_node = self.belief_graph.node_dict[room_id]
                unvisited_rooms_type.add(room_node.room_type)
        return list(unvisited_rooms_type)

    def get_unvisited_receptacles_type(self, room_id: int) -> set:
        unvisited_receptacles = set()
        for receptacle_id in self.receptacle_visited[room_id]:
            if not self.receptacle_visited[room_id][receptacle_id]:
                receptacle_node = self.belief_graph.node_dict[receptacle_id]
                unvisited_receptacles.add(receptacle_node.category)
        return list(unvisited_receptacles)

    def insert_to_room(self, lost_node: SceneNode) -> tuple[SceneNode, float]:
        node_category = lost_node.category
        unvisited_rooms = self.get_unvisited_rooms_type()
        assert len(unvisited_rooms) > 0
        room_type_result = self.agent.get_most_likely_room(
            node_category, unvisited_rooms
        )
        room_type = self.env.get_most_similar_room(room_type_result)
        rooms = self.belief_graph.get_room_by_type(room_type)
        index = random.randint(0, len(rooms) - 1)
        return rooms[index]  # TODO: if muti-room setting is necessary

    def insert_to_receptacle(
        self, lost_node: SceneNode, room_node: SceneNode
    ) -> SceneNode:
        node_category = lost_node.category
        unvisited_receptacles = self.get_unvisited_receptacles_type(room_node.id)
        assert len(unvisited_receptacles) > 0
        result = self.agent.get_most_likely_receptacle(
            node_category, room_node.room_type, unvisited_receptacles
        )
        receptacle_type_result = result
        receptacle_type = self.env.get_most_similar_receptacle(
            room_node.id, receptacle_type_result
        )
        receptacles = self.belief_graph.get_receptacle_by_type(
            room_node.id, receptacle_type
        )
        index = random.randint(0, len(receptacles) - 1)
        return receptacles[index]

    def insert_per_lost_node(self, lost_node_id: int):
        lost_node = self.task_graph.node_dict[lost_node_id]
        room_node = self.insert_to_room(lost_node)
        receptacle_node = self.insert_to_receptacle(lost_node, room_node)
        # add node and edge to the belif graph
        lost_node.set_to_estimation()
        self.belief_graph.add_node(lost_node)
        new_edge = SceneEdge(
            receptacle_node.id, lost_node.id, semantic_info={"relationType": "on"}
        )
        new_edge.add_relation(receptacle_node, lost_node)
        self.belief_graph.add_edge(new_edge)
        logger.info(
            f"[Estimate] insert {lost_node} to {receptacle_node} in {room_node}"
        )

    def insert_lost_nodes(self, lost_nodes_id: list[int]):
        for lost_node_id in lost_nodes_id:
            self.insert_per_lost_node(lost_node_id)

    def insert_exploration_action(self, rough_plan_step: Action):
        robot_location_node = self.belief_graph.robot_node_id
        if rough_plan_step.action_type == ActionType.Pick:
            _, child_id = rough_plan_step.del_edge
            # check if the robot is close to the action node
            receptacle_node = self.belief_graph.get_parent_receptacle(child_id)
            if receptacle_node.id != robot_location_node:
                insert_explore_action = Action(
                    action_type=ActionType.Walk,
                    edge_edit_pair=((receptacle_node.id, receptacle_node.id), None),
                )
                return False, insert_explore_action
        elif rough_plan_step.action_type == ActionType.Place:
            # check if the robot is close to the action node
            parent_id, child_id = rough_plan_step.add_edge
            if child_id in set(self.belief_graph.robot.nodes) and parent_id in set(
                self.belief_graph.robot.nodes
            ):
                logger.info(f"Skip: {child_id} has been placed in {parent_id}")
                return True, None
            receptacle_node = self.belief_graph.get_parent_receptacle(parent_id)
            if receptacle_node.id != robot_location_node:
                insert_explore_action = Action(
                    action_type=ActionType.Walk,
                    edge_edit_pair=((receptacle_node.id, receptacle_node.id), None),
                )
                return False, insert_explore_action
        return True, None

    def add_new_receptacle_node(self, node: SceneNode, edge: SceneEdge):
        self.belief_graph.add_node(node)
        self.belief_graph.add_edge(edge)
        self.task_graph.add_node(node)
        self.task_graph.add_edge(edge)
        room_id = edge.parent_id
        self.receptacle_visited[room_id][node.id] = True
        logger.info(f"add new receptacle node {node}")

    def update_belief_graph(self, obs: Observation, action: Action) -> bool:
        # update the belif graph with action
        update_graph(self.belief_graph, self.task_graph, action, self.env, False)

        # update visited map with the robot location
        robot_room_id = self.belief_graph.get_parent_room_id(
            self.belief_graph.robot_node_id
        )
        assert robot_room_id == obs.room_node.id
        logger.info(f"Robot in {obs.room_node}")
        for room_id in self.room_visited:
            if room_id == robot_room_id:
                self.room_visited[robot_room_id] = True
        # update the receptacle visited map
        for receptacle_id in self.receptacle_visited[robot_room_id]:
            if receptacle_id not in obs.closed_containers:
                self.receptacle_visited[robot_room_id][receptacle_id] = True

        # update the belif graph with observation
        replan_flag = False
        belif_nodes_id = set(self.belief_graph.graph.nodes)
        # in belif graph
        for node_id in belif_nodes_id:
            assert (
                node_id in self.task_graph.node_dict
                or self.belief_graph.node_dict[node_id].is_virtual()
            )
            belif_node = self.belief_graph.node_dict[node_id]
            if node_id not in self.belief_graph.edge_dict:
                continue
            belif_edge = self.belief_graph.edge_dict[node_id]
            if node_id in obs.node_visible:
                if not belif_node.is_truth:
                    obs_edge = obs.edge_visible[node_id]
                    obs_node = obs.node_visible[node_id]
                    if obs_edge == belif_edge:
                        belif_node.set_to_truth()
                        logger.info(f"update belif node {obs_node}")
                    else:
                        # update the node relation
                        self.belief_graph.update_egde(belif_edge, obs_edge)
                        self.belief_graph.update_node(belif_node, obs_node)
                        replan_flag = True
                        belif_node.set_to_truth()
                        logger.info(
                            f"update belif graph {obs_node}: {belif_edge} -> {obs_edge}"
                        )
                        if obs_edge.parent_id not in self.belief_graph.node_dict:
                            parent_node = obs.node_visible[obs_edge.parent_id]
                            parent_edge = obs.edge_visible[obs_edge.parent_id]
                            self.add_new_receptacle_node(parent_node, parent_edge)
            else:
                bnode_parent_id = self.belief_graph.get_parent_receptacle(node_id).id
                if (
                    bnode_parent_id in self.receptacle_visited[robot_room_id]
                    and self.receptacle_visited[robot_room_id][bnode_parent_id]
                ):
                    # re insert the node
                    logger.info(f"{belif_node} is not visible in {obs.room_node}")
                    self.belief_graph.remove_node(node_id)
                    self.insert_per_lost_node(node_id)
                    replan_flag = True
        return replan_flag

    def global_plan(self) -> list[Action]:
        action_seq, constraints = self.agent.generate_constraint_set(
            self.belief_graph, self.task_graph
        )
        problem = PlanningOnDynamicGraphProblem(
            self.belief_graph,
            self.task_graph,
            env=self.env,
            action_seq=action_seq,
            constraints=constraints,
        )
        min_cost_path, _ = test(
            Searcher, problem=problem, pruned=True, find_min_path=self.path_opt
        )
        logger.info("Global Planning")
        return path_to_action_sequence(min_cost_path)[1:]

    def local_plan(
        self, current_graph: SceneGraph, action_sequence: list[Action]
    ) -> None:
        for action in action_sequence:
            exception, message = self.simulator.simulate_step(
                graph=current_graph, action=action
            )
            if exception:
                logger.info(f"exception: {message}")
                message.parking_place = self.belief_graph.get_current_room_id()
                insert_actions = self.agent.generate_reslove_actions(message=message)
                logger.info(f"reslove plan: {insert_actions}")
                self.local_plan(current_graph.copy(), insert_actions)
            else:
                self.local_action_seq.append(action)

    def roll_out(self, action: Action) -> list[Action]:
        obs, is_skip = self.env.step(action)
        replan_flag = self.update_belief_graph(obs, action)
        if not is_skip:
            self.traj.append(action)
        return replan_flag

    def eval(self, visualize: bool = False) -> PlanningMetrics:
        moving_cost = 0
        sx, sy, sz = self.belief_graph.node_dict[self.env.respawn_node_id].position
        gsx, gsz = self.env.nav_map.convert_to_grid(sx, sz)
        real_path = [(sx, sz)]
        grid_path = [(gsx, gsz)]
        index = 0
        for action in self.traj:
            if action.action_type == ActionType.Walk:
                _, node_id = action.del_edge
                node = self.belief_graph.node_dict[node_id]
                x, y, z = node.position
                real_path.append((x, z))
                moving_cost += self.env.nav_map.distance(
                    real_path[index], real_path[index + 1]
                )
                index += 1
                gx, gz = self.env.nav_map.convert_to_grid(x, z)
                grid_path.append((gx, gz))
        if visualize:
            self.env.nav_map.show_grid_map(gsx, gsz, gsx, gsz, grid_path)
        node_expanded = 0
        for room_id in self.room_visited:
            if self.room_visited[room_id]:
                node_expanded += len(self.env.state_graph.get_subgraph(room_id))
        if self.logger:
            self.logger.info(self.traj)
        for action in self.traj:
            print_action(action, self.belief_graph)
        result = PlanningMetrics(
            moving_cost, node_expanded, len(self.traj), self.is_goal_achieved()
        )
        logger.info(f"Moving Cost: {moving_cost}")
        return result

    def is_goal_achieved(self) -> bool:
        _, _, _, edge_edit_pairs = ged_seq(self.belief_graph, self.task_graph)
        return len(edge_edit_pairs) == 0

    def main_loop(self):
        # get estimate belif graph
        self.init_visited_map()
        lost_nodes = self.get_lost_nodes()
        self.insert_lost_nodes(lost_nodes)
        # get estimate global plan ----> offline plan
        rough_plan = self.global_plan()
        # roll out the plan ----> online plan
        while len(rough_plan) > 0:
            # global replan
            rough_plan_step = rough_plan.pop(0)
            is_success, insert_explore_action = self.insert_exploration_action(
                rough_plan_step
            )
            if not is_success:
                rough_plan.insert(0, rough_plan_step)
                global_replan_flag = self.roll_out(insert_explore_action)
                if global_replan_flag:
                    rough_plan = self.global_plan()
            else:
                self.local_action_seq = []
                self.local_plan(self.belief_graph.copy(), [rough_plan_step])
                for action in self.local_action_seq:
                    global_replan_flag = self.roll_out(action)
                    if global_replan_flag:
                        rough_plan = self.global_plan()


if __name__ == "__main__":
    file_dir = "data/task/working_in_bedroom"
    env = ProcEnv(file_dir, 377)
    env.get_top_down_frame()
    epog = EPoG(env, global_agent="llm", local_agent="llm", path_opt=True)
    epog.main_loop()
    epog.eval(visualize=True)
