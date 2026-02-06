import json
import re

from loguru import logger

from epog.algorithm.efs import EFS
from epog.envs.graph import SceneGraph
from epog.utils.gpt_helper import ChatGPT
from pog.planning.action import Action, ActionType


class LLM_Explore(EFS):
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

    def create_action_from_raw(self, raw_action: str) -> Action:
        pattern = r"(\w+)\((\d+),\s*(\d+)\)"
        match = re.match(pattern, raw_action)
        logger.debug(f"Match: {match}")
        if match:
            action_type = match.group(1)
            object_id1 = int(match.group(2))
            object_id2 = int(match.group(3))

            if action_type == "Pick":
                action = Action(
                    action_type=ActionType.Pick,
                    edge_edit_pair=((object_id2, object_id1), None),
                )
            elif action_type == "Place":
                action = Action(
                    action_type=ActionType.Place,
                    edge_edit_pair=(None, (object_id2, object_id1)),
                )

            return action

        pattern = r"(\w+)\((\d+)\)"
        match = re.match(pattern, raw_action)
        if match:
            action_type = match.group(1)
            object_id1 = int(match.group(2))
            if action_type == "Open":
                action = Action(
                    action_type=ActionType.Open,
                    edge_edit_pair=((object_id1, object_id1), None),
                )
            elif action_type == "Close":
                action = Action(
                    action_type=ActionType.Close,
                    edge_edit_pair=((object_id1, object_id1), None),
                )
            elif action_type == "Walk":
                action = Action(
                    action_type=ActionType.Walk,
                    edge_edit_pair=((object_id1, object_id1), None),
                )
            return action

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

    # Overload
    def local_plan(self, current_graph: SceneGraph, action_sequence: list[Action]):
        for action in action_sequence:
            exception, message = self.simulator.simulate_step(
                graph=current_graph, action=action
            )
            return message

    def roll_out(self, action: Action) -> bool:
        obs, is_skip = self.env.step(action)
        if is_skip:
            return False

        self.update_belif_graph(obs, action)
        if not is_skip:
            self.traj.append(action)
        return True

    def get_plan(self, current_location, current_description, goal_description) -> None:
        model = ChatGPT()
        system_prompt = """You are an AI robot that generate a plan of actions to reach the goal.
                            The goal is to perform pick and place actions to move objects from one location to another in the housing scenario.
                            You will be given the initial state of the environment and the goal state.
                            The primitive actions are pick, place, open, close, and walk:
                            'Pick(x, y): Pick Object x from Object y'
                            'Place(x, y): Place Object x on Object y',
                            'Open(x): Open container x',
                            'Close(x): Close container x',
                            'Walk(x): Walk to location x'
                            The robot can only perform one action at a time.
                            The robot should walk to location y before performing Pick(x, y), Place(x, y)
                            The robot should walk to location x before performing Open(x), Close(x)
                            When the robot opens a container, it can see the objects inside the container.
                            If the robot don't know the location of an object, it should walk around and open containers to see where the object is.
                            x, y should be the id of the object or location.
                        """
        user_prompt = f"""The robot current location is {current_location}
                          The current state of the environment is given below.
                          The object name is followed by the id, e.g., 51 bread.
                          {current_description}
                          The goal state is given below.
                          {goal_description}
                          Please organize the output following the json format below:

                           "Plan":[
                                "Walk(5)",
                                "Pick(2, 5)",
                                "Walk(3)",
                                "Place(2, 3)",
                            ]
                        """
        response = model.get_response_text(system_prompt, user_prompt, None)
        json_llm = self.get_json_llm(response)
        logger.debug(f"JSON from LLM: {json_llm}")

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

    def main_loop(self):
        self.init_visited_map()
        current_description = self.output_belief_graph()
        goal_description = self.output_task_graph()
        current_location = self.belif_graph.robot_node_id
        logger.info(f"Current location: {self.belif_graph.node_dict[current_location]}")
        self.get_plan(current_location, current_description, goal_description)
        rough_plan = self._plan
        co = 0
        while (len(rough_plan) > 0 and co < 20) or (not self.is_goal_achieved()):
            try:
                rough_plan_step = rough_plan.pop(0)
            except IndexError:
                rough_plan = self.get_plan(
                    current_location, current_description, goal_description
                )
                co += 1
                continue
            is_success, _ = self.insert_exploration_action(rough_plan_step)
            if not is_success:
                logger.warning(f"Action not passing check1: {rough_plan_step}")
                current_description = self.output_belief_graph()
                goal_description = self.output_task_graph()
                current_location = self.belif_graph.robot_node_id
                self.get_plan(current_location, current_description, goal_description)
                rough_plan = self._plan
            else:
                logger.debug(f"Action passing check1: {rough_plan_step}")
                success = self.roll_out(rough_plan_step)
                if not success:
                    logger.warning(f"Action not passing check2: {rough_plan_step}")
                    current_description = self.output_belief_graph()
                    goal_description = self.output_task_graph()
                    current_location = self.belif_graph.robot_node_id

                    self.get_plan(
                        current_location, current_description, goal_description
                    )
                    rough_plan = self._plan


if __name__ == "__main__":
    from epog.envs.proc_env import ProcEnv

    file_dir = "data/task/prepare_breakfast"
    env = ProcEnv(file_dir, 69)
    llm = LLM_Explore(env)
    llm.main_loop()
    llm.eval(visualize=True)
