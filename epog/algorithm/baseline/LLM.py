import json
import re

import numpy as np
from loguru import logger
from python_tsp.exact import solve_tsp_dynamic_programming

from epog.algorithm.epog.EPoG import EPoG
from epog.envs.proc_env import Observation, ProcEnv, update_graph
from epog.envs.proc_scene import SceneNode
from epog.utils.gpt_helper import ChatGPT
from pog.planning.action import Action, ActionType


class LLM(EPoG):
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
        distance_room = {}
        for i in range(len(resorted_rooms)):
            for j in range(len(resorted_rooms)):
                room1 = self.belif_graph.room_nodes[i]
                room2 = self.belif_graph.room_nodes[j]
                distance = 0 if i == j else self.env.calculate_dis(room1.id, room2.id)
                distance_room[room1.id, room2.id] = distance
        distance_matrix = np.array(
            [
                [
                    distance_room[room1.id, room2.id]
                    for room2 in self.belif_graph.room_nodes
                ]
                for room1 in self.belif_graph.room_nodes
            ]
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

    def get_json_llm(self, llm_output):
        json_start = llm_output.find("{")
        json_end = llm_output.rfind("}")
        json_part = llm_output[json_start : json_end + 1]
        return json_part

    def is_valid_json(self, j_s):
        try:
            json.loads(j_s)
            return True
        except json.JSONDecodeError:
            return False

    def create_action_from_raw(self, raw_action: str) -> Action | None:
        pattern = r"(\w+)\((\d+)\s\w+,\s(\d+)\s\w+\)"
        match = re.match(pattern, raw_action)
        if not match:
            return None
        action_type = match.group(1)
        object_id2 = int(match.group(2))
        object_id1 = int(match.group(3))
        if action_type == "Pick":
            return Action(
                action_type=ActionType.Pick,
                edge_edit_pair=((object_id1, object_id2), None),
            )
        elif action_type == "Place":
            return Action(
                action_type=ActionType.Place,
                edge_edit_pair=(None, (object_id1, object_id2)),
            )
        return None

    def json_to_actions(self, json_data: dict[str, list]):
        raw_plan = json_data["Plan"]
        action_plan = []

        for raw_action in raw_plan:
            action = self.create_action_from_raw(raw_action)
            action_plan.append(action)

        return action_plan

    def output_belief_graph(self) -> str:
        output = ""
        for edge in self.belif_graph.edge_dict.values():
            name_item1 = self.belif_graph.node_dict[edge.child_id]
            name_item2 = self.belif_graph.node_dict[edge.parent_id]
            s = f"{name_item1} is {edge.semantic_info['relationType']} {name_item2}"
            output += s + "\n"
        return output

    def output_task_graph(self) -> str:
        output = ""
        for edge in self.task_graph.edge_dict.values():
            name_item1 = self.task_graph.node_dict[edge.child_id]
            name_item2 = self.task_graph.node_dict[edge.parent_id]
            s = f"{name_item1} is {edge.semantic_info['relationType']} {name_item2}"
            output += s + "\n"
        return output

    def main_loop(self):
        self.init_visited_map()
        room_exploration_seq = self.get_room_exploration_seq()
        logger.info("Exploring...")
        logger.info(f"Room exploration sequence: {room_exploration_seq}")
        for room in room_exploration_seq:
            self.explore_room(room)
            unexplored_containers = self.get_unexplored_containers(room.id)
            for container in unexplored_containers:
                self.explore_container(container)

        initial_description = self.output_belief_graph()
        goal_description = self.output_task_graph()

        model = ChatGPT()
        system_prompt = """You are an AI robot that generate a plan of actions to reach the goal.
                            The goal is to perform pick and place actions to move objects from one location to another in the housing scenario.
                            You will be given the initial state of the environment and the goal state.
                            The primitive actions are pick and place:
                            'Pick(x, y): Pick x from y'
                            'Place(x, y): Place x on y',
                        """
        user_prompt = f"""The initial state of the environment is given below.
                          The object name is followed by the id, e.g., 51 bread.
                          {initial_description}
                          The goal state is given below.
                          {goal_description}
                          Please organize the output following the json format below:

                           "Plan":[
                                "Pick(2 Box, 5 Garage)",
                                "Place(2 Box, 3 Kitchen)",
                            ]
                        """
        response = model.get_response_text(system_prompt, user_prompt, None)
        json_llm = self.get_json_llm(response)

        if not self.is_valid_json(json_llm):
            logger.warning("LLM output is not valid JSON")
            return None

        try:
            json_data = json.loads(json_llm)
        except json.JSONDecodeError:
            logger.warning("LLM output is not valid JSON")
            return None

        try:
            self._plan = self.json_to_actions(json_data)
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"LLM output is not valid action: {e}")
            return None

        logger.info("LLM-generated plan:")
        for action in self._plan:
            logger.info(f"  {action}")
        rough_plan = self._plan
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
    file_dir = "data/task/prepare_bath"
    env = ProcEnv(file_dir, 69)
    llm = LLM(env)
    llm.main_loop()
