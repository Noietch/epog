import numpy as np
from loguru import logger
from python_tsp.exact import solve_tsp_dynamic_programming

from epog.algorithm.epog.EPoG import EPoG
from epog.envs.proc_env import Observation, ProcEnv, update_graph
from epog.envs.proc_scene import SceneNode
from pog.planning.action import Action, ActionType


class EFS(EPoG):
    def update_belif_graph(self, obs: Observation, action: Action) -> bool:
        # update the belif graph with action
        update_graph(self.belif_graph, self.task_graph, action, self.env, False)
        # update visited map with the robot location
        robot_room_id = self.belif_graph.get_parent_room_id(
            self.belif_graph.robot_node_id
        )
        assert robot_room_id == obs.room_node.id
        logger.info(f"Robot in {obs.room_node}")
        for room_id in self.room_visited:
            if room_id == robot_room_id:
                self.room_visited[robot_room_id] = True
        # update the receptacle visited map
        robot_receptacle_id = self.belif_graph.get_parent_receptacle(
            self.belif_graph.robot_node_id
        ).id
        for receptacle_id in self.receptacle_visited[robot_room_id]:
            if receptacle_id not in obs.closed_containers:
                self.receptacle_visited[robot_room_id][receptacle_id] = True
        self.receptacle_visited[robot_room_id][robot_receptacle_id] = True
        # update the belif graph with observation
        replan_flag = False
        for node in obs.node_visible.values():
            if node.id not in self.belif_graph.node_dict:
                edge = obs.edge_visible[node.id]
                self.belif_graph.add_node(node)
                self.belif_graph.add_edge(edge)
                if node.id not in self.task_graph.node_dict:
                    self.task_graph.add_node(node)
                    self.task_graph.add_edge(edge)
        return replan_flag

    def get_room_exploration_seq(self) -> list[SceneNode]:
        # resorted rooms
        robot_node = self.belif_graph.robot_node_id
        resorted_rooms = [robot_node]
        id2room = {}
        count = 0
        for room in self.belif_graph.room_nodes:
            if room.id != robot_node:
                count += 1
                resorted_rooms.append(room.id)
                id2room[count] = room
            else:
                id2room[0] = room
        # distance matrix
        distance_matrix = []
        for i in resorted_rooms:
            for j in resorted_rooms:
                distance = self.env.calculate_dis(i, j)
                distance_matrix.append(distance)
        distance_matrix = np.array(distance_matrix).reshape(
            len(resorted_rooms), len(resorted_rooms)
        )
        distance_matrix[:, 0] = 0
        permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
        # get the room sequence
        room_seq = [id2room[room_id] for room_id in permutation]
        return room_seq

    def get_unexplored_containers(self, room_id: int):
        unexplored_containers = []
        for receptacle in self.belif_graph.receptacle_nodes[room_id]:
            if not self.receptacle_visited[room_id][receptacle.id]:
                unexplored_containers.append(receptacle)
        return unexplored_containers

    def explore_room(self, room_node: SceneNode):
        walk_action = Action(
            action_type=ActionType.Walk,
            edge_edit_pair=((room_node.id, room_node.id), None),
        )
        self.roll_out(walk_action)

    def explore_container(self, container_node: SceneNode):
        walk_action = Action(
            action_type=ActionType.Walk,
            edge_edit_pair=((container_node.id, container_node.id), None),
        )
        open_action = Action(
            action_type=ActionType.Open,
            edge_edit_pair=((container_node.id, container_node.id), None),
        )
        close_action = Action(
            action_type=ActionType.Close,
            edge_edit_pair=((container_node.id, container_node.id), None),
        )
        for action in [walk_action, open_action, close_action]:
            self.roll_out(action)

    def can_plan(self):
        belif_nodes = set(self.belif_graph.node_dict.keys())
        task_nodes = set(self.task_graph.node_dict.keys())
        return belif_nodes == task_nodes

    def main_loop(self):
        self.init_visited_map()
        room_exploration_seq = self.get_room_exploration_seq()
        for room in room_exploration_seq:
            self.explore_room(room)
            unexplored_containers = self.get_unexplored_containers(room.id)
            for container in unexplored_containers:
                self.explore_container(container)
            if self.can_plan():
                logger.info("Finished map construction")
                break
        rough_plan = self.global_plan()
        while len(rough_plan) > 0:
            # global replan
            rough_plan_step = rough_plan.pop(0)
            is_success, insert_explore_action = self.insert_exploration_action(
                rough_plan_step
            )
            if not is_success:
                rough_plan.insert(0, rough_plan_step)
                global_replan_flag = self.roll_out(insert_explore_action)
                assert not global_replan_flag
            else:
                logger.info("Local replan")
                self.local_action_seq = []
                self.local_plan(self.belif_graph.copy(), [rough_plan_step])
                for action in self.local_action_seq:
                    global_replan_flag = self.roll_out(action)
                    if global_replan_flag:
                        rough_plan = self.global_plan()


if __name__ == "__main__":
    file_dir = "data/task/prepare_breakfast"
    env = ProcEnv(file_dir, 238)
    efs = EFS(env)
    efs.main_loop()
    efs.eval(visualize=True)
