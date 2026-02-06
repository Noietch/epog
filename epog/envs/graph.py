import copy
import json
from dataclasses import dataclass

import networkx as nx

from pog.graph.edge import Edge
from pog.graph.graph import Graph
from pog.graph.node import Node
from pog.graph.shape import Shape, ShapeType


@dataclass
class MotionInfo:
    block_nodes_id: list[int]
    collision_nodes_id: list[int]
    unstable_nodes_id: list[int]

    def to_dict(self):
        return {
            "block_nodes_id": self.block_nodes_id,
            "collision_nodes_id": self.collision_nodes_id,
            "unstable_nodes_id": self.unstable_nodes_id,
        }

    @staticmethod
    def from_dict(info: dict):
        return MotionInfo(
            info["block_nodes_id"],
            info["collision_nodes_id"],
            info["unstable_nodes_id"],
        )

    def involve_nodes_id(self):
        return self.block_nodes_id + self.collision_nodes_id + self.unstable_nodes_id


ASSETS_DB = "epog/envs/assets/asset2info.json"


class SceneNode(Node):
    def __init__(
        self,
        id: int,
        shape: Shape,
        semantic_info=None,
        motion_info=None,
        state=None,
        is_truth=True,
        **kwargs,
    ) -> None:
        super().__init__(id, shape, semantic_info, motion_info, **kwargs)
        self.is_truth = is_truth
        self.state = state
        if motion_info:
            if isinstance(motion_info, MotionInfo):
                self.motion_info = motion_info
            elif isinstance(motion_info, dict):
                self.motion_info = MotionInfo.from_dict(motion_info)
            else:
                raise ValueError("motion_info must be MotionInfo or dict")
        else:
            self.motion_info = MotionInfo([], [], [])

    def is_room(self) -> bool:
        return self.category == "room"

    def __repr__(self) -> str:
        if self.is_room():
            return str(self.id) + " " + " " + self.room_type
        return str(self.id) + " " + self.category

    @property
    def category(self) -> str:
        return self.semantic_info["category"]

    @property
    def room_type(self) -> str:
        if self.is_room():
            return self.semantic_info["roomType"]

    @property
    def position(self) -> tuple:
        x = self.semantic_info["position"]["x"]
        y = self.semantic_info["position"]["y"]
        z = self.semantic_info["position"]["z"]
        return (x, y, z)

    def set_to_truth(self):
        self.is_truth = True

    def set_to_estimation(self):
        self.is_truth = False

    def __eq__(self, other) -> bool:
        return self.id == other.id

    def is_virtual(self):
        return self.category == "Blocks"


class SceneEdge(Edge):
    def __init__(self, parent_id: int, child_id: int, **kwargs) -> None:
        super().__init__(parent_id, child_id, **kwargs)

    def __repr__(self) -> str:
        return f"Edge from {self.parent_id} to {self.child_id}"

    def __eq__(self, other) -> bool:
        if isinstance(other, SceneEdge):
            return (
                self.parent_id == other.parent_id
                and self.child_id == other.child_id
                and self.relation_type == other.relation_type
            )
        return False

    @property
    def relation_type(self):
        return self.semantic_info["relationType"]

    def get_affordance(self, node: SceneNode, is_parent: bool) -> dict:
        if node.shape.object_type == ShapeType.ARTIC:
            if is_parent:
                return node.affordance["cabinet_inner_middle"]
            else:
                return node.affordance["cabinet_outer_bottom"]
        else:
            if is_parent:
                return node.affordance["box_aff_pz"]
            else:
                return node.affordance["box_aff_nz"]

    def add_relation(
        self, parent_node: SceneNode, child_node: SceneNode, dof_type="x-y", pose=None
    ):
        if pose is None:
            pose = [0, 0, 0]
        parent_aff = self.get_affordance(parent_node, is_parent=True)
        child_aff = self.get_affordance(child_node, is_parent=False)
        return super().add_relation(parent_aff, child_aff, dof_type, pose)


class SceneGraph(Graph):
    def __init__(
        self,
        scene_name,
        scene_file=None,
        file_dir=None,
        file_name=None,
        fn=None,
        robot_node_id=None,
        **kwargs,
    ) -> None:
        super().__init__(scene_name, file_dir, file_name, fn, robot_node_id, **kwargs)

        # convert node_dict and edge_dict to SceneNode and SceneEdge
        self.node_dict: dict[int, SceneNode] = {
            k: SceneNode(
                id=v.id,
                shape=v.shape,
                semantic_info=v.semantic_info,
                motion_info=v.motion_info,
                state=v.state,
                is_truth=v.is_truth,
            )
            for k, v in self.node_dict.items()
        }
        self.edge_dict: dict[int, SceneEdge] = {
            k: SceneEdge(
                parent_id=v.parent_id,
                child_id=v.child_id,
                relations=v.relations,
                containment=v.containment,  # pog in/on
                parent_to_child_tf=v.parent_to_child_tf,
                semantic_info=v.semantic_info,
            )
            for k, v in self.edge_dict.items()
        }

        self.file_dir = file_dir
        with open(ASSETS_DB) as f:
            self.asset2info = json.load(f)
        self.scene_name = scene_name
        self.scene_file = scene_file
        self.room_nodes: list[SceneNode] = self.get_room_nodes()
        self.receptacle_nodes: dict[int, list[SceneNode]] = self.get_receptacle_nodes()
        self.controller = None

    def update_egde(self, old_edge: SceneEdge, new_edge: SceneEdge):
        self.graph.remove_edge(old_edge.parent_id, old_edge.child_id)
        self.graph.add_edge(new_edge.parent_id, new_edge.child_id, edge=new_edge)
        self.edge_dict[new_edge.child_id] = new_edge

    def update_node(self, old_node: SceneNode, new_node: SceneNode):
        assert old_node.id == new_node.id
        self.node_dict[old_node.id] = new_node

    def get_room_nodes(self) -> list[SceneNode]:
        room_nodes = []
        for node in self.node_dict.values():
            if node.category == "room":
                room_nodes.append(node)
        return room_nodes

    def get_parent_room_id(self, node_id: int):
        cur_node = self.node_dict[node_id]
        while not cur_node.is_room():
            cur_parent_id = self.edge_dict[cur_node.id].parent_id
            cur_node = self.node_dict[cur_parent_id]
        return cur_node.id

    def get_room_by_type(self, room_type: str) -> list[SceneNode]:
        rooms = []
        for node in self.room_nodes:
            if node.room_type.lower() == room_type.lower():
                rooms.append(node)
        return rooms

    def get_receptacle_nodes(self) -> dict[int, SceneNode]:
        receptacle_nodes = {node.id: [] for node in self.room_nodes}
        for node_id in receptacle_nodes:
            for child_id in self.graph.successors(node_id):
                node = self.node_dict[child_id]
                if "assetId" in node.semantic_info:
                    assetId = node.semantic_info["assetId"]
                    asset_info = self.asset2info[assetId]
                    if (
                        "secondaryProperties" in asset_info
                        and "Receptacle" in asset_info["secondaryProperties"]
                    ):
                        receptacle_nodes[node_id].append(node)
        return receptacle_nodes

    def get_receptacle_by_type(self, room_id: int, receptacle_type: str) -> SceneNode:
        receptacles = []
        for node in self.receptacle_nodes[room_id]:
            if node.category == receptacle_type:
                receptacles.append(node)
        return receptacles

    def get_ancestors(self, node_id: int):
        ancestors = [node_id]
        while node_id in self.edge_dict:
            node_id = self.edge_dict[node_id].parent_id
            ancestors.append(node_id)
        return ancestors

    def get_base_nodes_id(self) -> list[int]:
        nodes_id = []
        for node_id, receptacles in self.receptacle_nodes.items():
            nodes_id.append(node_id)
            for receptacle in receptacles:
                nodes_id.append(receptacle.id)
        return nodes_id + [self.root] + self.get_virtual_nodes_id()

    def get_parent_receptacle(self, node_id: int) -> SceneNode:
        if node_id == self.root:
            return self.node_dict[self.robot_node_id]
        cur_node = self.node_dict[node_id]
        if cur_node.is_room():
            return cur_node
        cur_parent = self.get_parent(cur_node)
        while not cur_parent.is_room():
            cur_node = cur_parent
            cur_parent = self.get_parent(cur_node)
        return cur_node

    def get_parent_room(self, node_id: int) -> SceneNode:
        cur_node = self.node_dict[node_id]
        while not cur_node.is_room():
            cur_node = self.get_parent(cur_node)
        return cur_node

    def get_current_room_id(self) -> int:
        return self.get_parent_room(self.robot_node_id).id

    def get_distance(self, object_id1: int, object_id2: int, dis_fn) -> float:
        parent_1 = self.get_parent_receptacle(object_id1)
        parent_2 = self.get_parent_receptacle(object_id2)
        distance = dis_fn(parent_1.id, parent_2.id)
        return distance, parent_1.id, parent_2.id

    def copy(self, include_nodes=None) -> "SceneGraph":
        def fn(include_nodes):
            node_dict = {}
            edge_dict = {}
            nodes_id = include_nodes
            if include_nodes is None:
                nodes_id = self.node_dict.keys()
            for item in nodes_id:
                node_dict[item] = copy.deepcopy(self.node_dict[item])
                if item != self.root and item in self.edge_dict:
                    edge_dict[item] = copy.deepcopy(self.edge_dict[item])
            root_id = self.root
            return node_dict, edge_dict, root_id

        graph_copy = SceneGraph(
            f"{self.scene_name} copy",
            scene_file=self.scene_file,
            file_dir=self.file_dir,
            fn=fn,
            include_nodes=include_nodes,
        )
        graph_copy.root_aff = self.root_aff
        graph_copy.robot_node_id = self.robot_node_id

        # copy robot tree (discard codes below if robot is not needed)
        if self.robot_root is not None:
            graph_copy.robot = self.robot.copy()
            graph_copy.robot_root = self.robot_root
            for item in self.robot.nodes:
                graph_copy.node_dict[item] = copy.deepcopy(self.node_dict[item])
            for item in self.robot.edges:
                if item != self.robot_root:
                    graph_copy.edge_dict[item[1]] = copy.deepcopy(
                        self.edge_dict[item[1]]
                    )
        return graph_copy

    def get_parent(self, node: SceneNode) -> SceneNode:
        return self.node_dict[self.edge_dict[node.id].parent_id]

    def save_gexf(self, save_path: str):
        G = nx.DiGraph()
        for node in self.node_dict.values():
            info = node.semantic_info
            category = info["id"].split("|")[0] if "id" in info else "house"
            if category == "room":
                category = info["roomType"]
            G.add_node(node.id, category=category)
        for edge in self.edge_dict.values():
            G.add_edge(edge.child_id, edge.parent_id)
        nx.write_gexf(G, save_path)

    def remove_virtual_nodes(self, node_id: int) -> None:
        for node in self.node_dict.values():
            if node_id in node.motion_info.collision_nodes_id:
                node.motion_info.collision_nodes_id.remove(node_id)
            if node_id in node.motion_info.unstable_nodes_id:
                node.motion_info.unstable_nodes_id.remove(node_id)
            if node_id in node.motion_info.block_nodes_id:
                node.motion_info.block_nodes_id.remove(node_id)
        self.graph.remove_node(node_id)

    def get_virtual_nodes_id(self) -> list[SceneNode]:
        virtual_nodes_id = []
        for node in self.node_dict.values():
            if node.is_virtual():
                virtual_nodes_id.append(node.id)
        return virtual_nodes_id
