import random

from epog.algorithm.epog.fake_simulator import MotionError, MotionErrorType
from epog.algorithm.llm_prompt import (
    get_most_likely_receptacle,
    get_most_likely_room,
    get_resolve_action_seq,
)
from epog.envs.graph import SceneGraph
from epog.utils.gpt_helper import ChatGPT
from pog.planning.action import Action, ActionConstraints, ActionType, ged_seq


class Planner:
    def __init__(
        self,
        global_agent: str = "expert",
        local_agent: str = "llm",
        state_graph: SceneGraph = None,
    ) -> None:
        self.global_expert = global_agent == "expert"
        self.local_expert = local_agent == "expert"
        self.state_graph = state_graph
        self.llm = ChatGPT()

    def generate_constraint_set(self, g1: SceneGraph, g2: SceneGraph) -> list[Action]:
        seq_from_ged = ged_seq(g1, g2)
        return self.pick_place_constraints(seq_from_ged[-1])

    def generate_reslove_actions(self, message: MotionError) -> list[Action]:
        if self.local_expert:
            return self.rule_based_replanner(message)
        else:
            return self.llm_based_replanner(message)

    def get_most_likely_room(self, object_category: str, room_list: list[str]) -> str:
        if self.global_expert:
            for room_name in room_list:
                if room_name == "Kitchen":
                    return room_name
            for room_name in room_list:
                return room_name
        else:
            return get_most_likely_room(object_category, room_list=room_list)

    def get_most_likely_receptacle(
        self, object_category: str, room: str, receptcle: list[str]
    ) -> str:
        if self.global_expert:
            receptcle = receptcle.copy()
            random.shuffle(receptcle)
            return receptcle[0]
        else:
            return get_most_likely_receptacle(object_category, room, receptcle)

    def llm_based_replanner(self, message: MotionError) -> list[Action]:
        return get_resolve_action_seq(message)

    def pick_place_constraints(self, seq_from_ged) -> list[Action]:
        action_seq = []
        constraints = ActionConstraints()
        object_in_hand_placement = []
        for edge_pair in seq_from_ged:
            if edge_pair[0] is None:
                add_action = Action((None, edge_pair[1]), action_type=ActionType.Place)
                object_in_hand_placement.append(add_action)
                action_seq.append(add_action)
            else:
                del_action = Action((edge_pair[0], None), action_type=ActionType.Pick)
                add_action = Action((None, edge_pair[1]), action_type=ActionType.Place)
                action_seq.append(del_action)
                action_seq.append(add_action)
                constraints.addConstraint(del_action, add_action)
        for place_action in object_in_hand_placement:
            for action in action_seq:
                if action in object_in_hand_placement:
                    continue
                constraints.addConstraint(place_action, action)
        return list(set(action_seq)), constraints

    def rule_based_replanner(self, message: MotionError) -> list[Action]:
        if message.error_type == MotionErrorType.AccessError:
            insert_actions = []
            for node_id in message.involved_nodes_id:
                open_action = Action(
                    ((node_id, node_id), (node_id, node_id)),
                    action_type=ActionType.Open,
                )
                insert_actions.append(open_action)
            insert_actions.append(message.failure_action)
            for node_id in message.involved_nodes_id:
                close_action = Action(
                    ((node_id, node_id), (node_id, node_id)),
                    action_type=ActionType.Close,
                )
                insert_actions.append(close_action)
        elif message.error_type == MotionErrorType.BlockError:
            insert_actions = []
            parent_id, child_id = message.failure_action.del_edge
            for node_id in message.involved_nodes_id:
                insert_actions.append(
                    Action(((parent_id, node_id), None), action_type=ActionType.Pick)
                )
                insert_actions.append(
                    Action(
                        (None, (message.parking_place, node_id)),
                        action_type=ActionType.Place,
                    )
                )
            insert_actions.append(message.failure_action)
        elif message.error_type == MotionErrorType.StabilityError:
            insert_actions = []
            parent_id, child_id = message.failure_action.del_edge
            for node_id in message.involved_nodes_id:
                insert_actions.append(
                    Action(((child_id, node_id), None), action_type=ActionType.Pick)
                )
                insert_actions.append(
                    Action(
                        (None, (message.parking_place, node_id)),
                        action_type=ActionType.Place,
                    )
                )
            insert_actions.append(message.failure_action)
        elif message.error_type == MotionErrorType.CollisionError:
            insert_actions = []
            parent_id, child_id = message.failure_action.add_edge
            for node_id in message.involved_nodes_id:
                insert_actions.append(
                    Action(((parent_id, node_id), None), action_type=ActionType.Pick)
                )
                insert_actions.append(
                    Action(
                        (None, (message.parking_place, node_id)),
                        action_type=ActionType.Place,
                    )
                )
            insert_actions.append(message.failure_action)
        return insert_actions
