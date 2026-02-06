import copy
import json
import os
from dataclasses import dataclass

import cv2
import networkx as nx
import numpy as np
import torch
from ai2thor.controller import Controller
from loguru import logger
from sentence_transformers import SentenceTransformer

from epog.envs.graph import SceneEdge, SceneGraph, SceneNode
from epog.envs.navigation import NavigationMap
from pog.graph.node import ContainmentState
from pog.planning.action import Action, ActionType


def update_graph(
    current: SceneGraph,
    goal: SceneGraph,
    action: Action,
    env: "ProcEnv",
    in_search: bool = True,
) -> tuple[float, bool]:
    moving_cost = 0
    parent_id, child_id = None, None
    if action.action_type == ActionType.Pick:
        parent_id, child_id = action.del_edge
        child_node = current.node_dict[child_id]
        if child_node.is_virtual():  # remove exception node
            current.remove_virtual_nodes(child_id)
            return moving_cost, False
        if current.edge_dict[child_id].parent_id != parent_id:
            logger.warning(f"Skip: position of node {parent_id} has been updated")
            return moving_cost, True
        moving_cost, _, child_receptacle = current.get_distance(
            current.robot_node_id, parent_id, env.calculate_dis
        )
        current.removeNode(child_id)
    elif action.action_type == ActionType.Place:
        parent_id, child_id = action.add_edge
        child_node = current.node_dict[child_id]
        parent_node = current.node_dict[parent_id]
        if child_node.is_virtual() or parent_node.is_virtual():  # skip exception node
            return moving_cost, False
        if child_id in set(current.robot.nodes) and parent_id in set(
            current.robot.nodes
        ):
            logger.warning(
                f"Skip: {child_id} not in current.node_dict and {parent_id} not in current.node_dict it is in robot hand"
            )
            return moving_cost, True
        if (
            child_id in current.edge_dict
            and current.edge_dict[child_id].parent_id == parent_id
        ):
            logger.warning(f"Skip: {child_id} has been placed in {parent_id}")
            return moving_cost, True
        moving_cost, _, child_receptacle = current.get_distance(
            current.robot_node_id, parent_id, env.calculate_dis
        )
        if goal.edge_dict[child_id].parent_id != parent_id:
            node_child = current.node_dict[child_id]
            node_parent = current.node_dict[parent_id]
            new_edge = SceneEdge(
                parent_id, child_id, semantic_info={"relationType": "on"}
            )
            new_edge.add_relation(node_parent, node_child)
            current.add_edge(new_edge)
        else:
            current.addNode(parent_id, edge=goal.edge_dict[child_id])
    elif action.action_type == ActionType.Walk:
        _, child_id = action.del_edge
        moving_cost, _, child_receptacle = current.get_distance(
            current.robot_node_id, child_id, env.calculate_dis
        )
        current.robot_node_id = child_receptacle
    elif action.action_type == ActionType.Open:
        _, child_id = action.del_edge
        current.node_dict[child_id].state = ContainmentState.Opened
    elif action.action_type == ActionType.Close:
        _, child_id = action.del_edge
        current.node_dict[child_id].state = ContainmentState.Closed
    else:
        raise ValueError("Invalid action type")
    # if in search walk action will not be inserted
    if in_search:
        current.robot_node_id = child_receptacle
    return moving_cost, False


def print_action(action: Action, graph: SceneGraph) -> None:
    if action.action_type == ActionType.Pick:
        child_node = graph.node_dict[action.del_edge[1]]
        parent_node = graph.node_dict[action.del_edge[0]]
        logger.info(f"Pick {child_node} from {parent_node}")
    elif action.action_type == ActionType.Place:
        child_node = graph.node_dict[action.add_edge[1]]
        parent_node = graph.node_dict[action.add_edge[0]]
        logger.info(f"Place {child_node} to {parent_node}")
    elif action.action_type == ActionType.Walk:
        parent_node = graph.node_dict[action.del_edge[0]]
        logger.info(f"Walk to {parent_node}")
    elif action.action_type == ActionType.Open:
        parent_node = graph.node_dict[action.del_edge[0]]
        logger.info(f"Open {parent_node}")
    elif action.action_type == ActionType.Close:
        parent_node = graph.node_dict[action.del_edge[0]]
        logger.info(f"Close {parent_node}")
    else:
        raise ValueError("Invalid action type")


@dataclass
class Observation:
    room_node: SceneNode
    node_visible: dict[int, SceneNode]
    edge_visible: dict[int, SceneEdge]
    closed_containers: list[int]


init_transformer = None
if init_transformer is None:
    _transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


class ProcEnv:
    def __init__(self, file_dir: str, scene_id: int) -> None:
        # load task env
        self.extended_file_dir = os.path.join(file_dir, f"scene_{scene_id}")
        with open(
            os.path.join(self.extended_file_dir, f"{scene_id}_scene.json")
        ) as f:
            self.scene = json.load(f)
        self.state_graph = SceneGraph(
            "state_graph",
            self.scene,
            self.extended_file_dir,
            file_name=f"{scene_id}_start.json",
        )
        self.task_graph = SceneGraph(
            "task_graph",
            self.scene,
            self.extended_file_dir,
            file_name=f"{scene_id}_task.json",
        )
        self.goal_graph = SceneGraph(
            "goal_graph",
            self.scene,
            self.extended_file_dir,
            file_name=f"{scene_id}_goal.json",
        )
        with open(
            os.path.join(self.extended_file_dir, f"{scene_id}_task_info.json")
        ) as f:
            self.task_info = json.load(f)

        # init the environment
        self.transformer = _transformer
        self.node2level = self.get_node2level()
        self.room_embedding, self.receptacle_embedding = self.init_embedding()

        # init the robot
        self.respawn()

        # init the navigation map
        self.controller = Controller(scene=self.scene)
        self.room_nodes: list[SceneNode] = self.state_graph.get_room_nodes()
        self.nav_map = NavigationMap(self.controller)
        self.temp_distance_map = {}

    def init_embedding(self) -> np.ndarray:
        logger.info("Init embedding...")
        embedding_path = os.path.join(self.extended_file_dir, "embedding.pt")
        if os.path.exists(embedding_path):
            embedding = torch.load(embedding_path, weights_only=False)
            return embedding["room"], embedding["receptacle"]
        # get room embedding
        room_types = [node.room_type for node in self.state_graph.room_nodes]
        room_embedding = {
            "types": room_types,
            "embedding": self.transformer.encode(room_types, normalize_embeddings=True),
        }
        # get receptacle embedding
        receptacle_embedding = {}
        for room_id, receptacles in self.state_graph.receptacle_nodes.items():
            receptacle_types = []
            for receptacle in receptacles:
                receptacle_types.append(receptacle.category)
            receptacle_embedding[room_id] = {
                "types": receptacle_types,
                "embedding": self.transformer.encode(
                    receptacle_types, normalize_embeddings=True
                ),
            }
        torch.save(
            {"room": room_embedding, "receptacle": receptacle_embedding}, embedding_path
        )
        logger.info("Init embedding done.")
        return room_embedding, receptacle_embedding

    def get_top_down_frame(self) -> np.ndarray:
        # Setup the top-down camera
        event = self.controller.step(
            action="GetMapViewCameraProperties", raise_for_failure=True
        )
        pose = copy.deepcopy(event.metadata["actionReturn"])

        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 50
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = False
        pose["farClippingPlane"] = 50
        del pose["orthographicSize"]

        # add the camera to the scene
        event = self.controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
        top_down_frame = event.third_party_camera_frames[-1]
        cv2.imwrite("top_down_frame.png", top_down_frame)
        return top_down_frame

    def get_most_similar(
        self, query: str, corpus: list[str], embedding: np.ndarray
    ) -> str:
        query_embedding = self.transformer.encode(query, normalize_embeddings=True)
        similarity = self.transformer.similarity(query_embedding, embedding)
        index = np.argmax(similarity)
        return corpus[index]

    def get_most_similar_room(self, room_type: str) -> SceneNode:
        # the most similar room type
        return self.get_most_similar(
            room_type, self.room_embedding["types"], self.room_embedding["embedding"]
        )

    def get_most_similar_receptacle(
        self, room_id: int, receptacle_type: str
    ) -> list[SceneNode]:
        return self.get_most_similar(
            receptacle_type,
            self.receptacle_embedding[room_id]["types"],
            self.receptacle_embedding[room_id]["embedding"],
        )

    def respawn(self) -> None:
        include_room = self.task_info["include_room"]
        for room in self.state_graph.room_nodes:
            if room.room_type not in include_room:
                self.respawn_node_id = room.id
                self.state_graph.respwan_robot(room.id)
                break
        logger.info(f"Respawn at {room}")

    def get_node2level(self) -> dict[int, int]:
        return nx.single_source_shortest_path_length(
            self.state_graph.graph, source=self.state_graph.root
        )

    def calculate_dis(
        self, object_id1: int, object_id2: int, show_animation=False
    ) -> float:
        # use temp memory
        if (
            object_id1 in self.temp_distance_map
            and object_id2 in self.temp_distance_map[object_id1]
        ):
            return self.temp_distance_map[object_id1][object_id2]
        if (
            object_id2 in self.temp_distance_map
            and object_id1 in self.temp_distance_map[object_id2]
        ):
            return self.temp_distance_map[object_id2][object_id1]

        position1 = self.state_graph.node_dict[object_id1].semantic_info["position"]
        position2 = self.state_graph.node_dict[object_id2].semantic_info["position"]
        start_x, start_y = position1["x"], position1["z"]
        goal_x, goal_y = position2["x"], position2["z"]
        moving_cost = self.nav_map.distance(
            (start_x, start_y), (goal_x, goal_y), show_animation
        )
        # temp memory
        if object_id1 not in self.temp_distance_map:
            self.temp_distance_map[object_id1] = {}
        self.temp_distance_map[object_id1][object_id2] = moving_cost
        return moving_cost

    # TODO: add the container constraint
    def get_observations(
        self,
    ) -> tuple[list[SceneNode], dict[int, SceneEdge], list[int]]:
        # get self.state_graph room
        character_id = self.state_graph.robot_node_id
        cur_room_id = self.state_graph.get_parent_room_id(character_id)
        cur_room = self.state_graph.node_dict[cur_room_id]

        # get all nodes and edges inside the room
        nodes_id: list[int] = self.state_graph.get_subgraph(cur_room_id)

        # get all nodes that are visible to the character
        node_visible = {cur_room_id: cur_room}
        closed_containers = set()
        for node_id in nodes_id:
            node = self.state_graph.node_dict[node_id]
            parent_node = self.state_graph.get_parent_receptacle(node_id)
            if (
                parent_node.id != node_id
                and parent_node.state == ContainmentState.Closed
            ):
                continue
            if node.state == ContainmentState.Closed:
                closed_containers.add(node_id)
            node_visible[node_id] = node

        # get all edges that are visible to the character
        edge_visible = {}
        for edge in self.state_graph.edge_dict.values():
            if edge.child_id in node_visible and edge.parent_id in node_visible:
                edge_visible[edge.child_id] = edge

        return Observation(
            cur_room, node_visible, edge_visible, list(closed_containers)
        )

    def get_init_graph(self) -> SceneGraph:
        node_bases = self.state_graph.get_base_nodes_id()
        return self.state_graph.copy(include_nodes=node_bases)

    def step(self, action: Action) -> tuple[Observation, bool]:
        print_action(action, self.state_graph)
        _, is_skip = update_graph(
            self.state_graph, self.goal_graph, action, self, False
        )
        observations = self.get_observations()
        return observations, is_skip

    def close(self) -> None:
        self.controller.stop()
        self.controller = None
        del self.transformer
