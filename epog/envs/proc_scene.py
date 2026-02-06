import copy
import json

import cv2
import networkx as nx
import numpy as np
import trimesh
import vedo
from ai2thor.controller import Controller

from epog.envs.graph import SceneEdge, SceneGraph, SceneNode
from pog.graph.shape import Box
from pog.graph.shapes import ComplexStorage

ASSETS_DB = "epog/envs/assets/asset2info.json"


class ProcScene:
    default_container_list = ["Cabinet", "Fridge", "Microwave"]

    def __init__(self, house: dict, quiet: bool = False) -> None:
        self.house = house
        self.textid2id()
        with open(ASSETS_DB) as f:
            self.assets_db = json.load(f)
        if not quiet:
            self.controller = Controller(scene=self.house)
            self.normal_graph = self.parse_graph()

    def textid2id(self) -> None:
        self.id2obj = {}
        self.obj2id = {}
        cur_id = 1
        self.id2obj[self.house["metadata"]["roomSpecId"]] = 0
        self.obj2id[0] = self.house["metadata"]["roomSpecId"]
        for obj in self.house["rooms"]:
            self.id2obj[obj["id"]] = cur_id
            self.obj2id[cur_id] = obj["id"]
            cur_id += 1
        for obj in self.house["objects"]:
            self.id2obj[obj["id"]] = cur_id
            self.obj2id[cur_id] = obj["id"]
            cur_id += 1
            if "children" in obj:
                for child in obj["children"]:
                    self.id2obj[child["id"]] = cur_id
                    self.obj2id[cur_id] = obj["id"]
                    cur_id += 1

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

    def conver_to_mesh(self) -> None:
        scene = trimesh.Scene()
        for node in self.house["objects"]:
            nodes = [node]
            if "children" in node:
                nodes.extend(node["children"])
            for obj in nodes:
                asset_geometry = self.controller.step(
                    action="GetInSceneAssetGeometry",
                    objectId=obj["id"],
                    triangles=True,
                    renderImage=False,
                ).metadata["actionReturn"]
                for _j, mesh_info in enumerate(asset_geometry):
                    # NOTE: Swaps y and z dimensions
                    vertices = np.array(
                        [[p["x"], p["z"], p["y"]] for p in mesh_info["vertices"]]
                    )
                    triangles = np.array(mesh_info["triangles"]).reshape(-1, 3)[
                        :, [0, 2, 1]
                    ]
                    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                    mesh.visual.face_colors = np.random.randint(0, 255, size=(1, 3))[
                        0, :
                    ]
                    scene.add_geometry(mesh)
        ploter = vedo.Plotter()
        ploter.add(scene.dump(concatenate=True))
        ploter.show(axes=1).close()

    def parse_graph(self, exclude_ids=None, exclude_categories=None) -> nx.DiGraph:
        if exclude_categories is None:
            exclude_categories = []
        if exclude_ids is None:
            exclude_ids = []
        graph = nx.DiGraph()
        # add the house
        self.house["metadata"]["category"] = "house"
        graph.add_node(
            self.house["metadata"]["roomSpecId"],
            category="house",
            object=self.house["metadata"],
        )
        # add rooms
        for room in self.house["rooms"]:
            if room["id"] in exclude_ids:
                continue
            room_type = room["roomType"]
            room["category"] = "room"
            graph.add_node(room["id"], category=room_type, object=room)
            graph.add_edge(self.house["metadata"]["roomSpecId"], room["id"])

        for obj in self.house["objects"]:
            room_id = obj["id"].split("|")[1]
            category = obj["id"].split("|")[0]
            if obj["id"] in exclude_ids or category in exclude_categories:
                continue
            obj["category"] = category
            graph.add_node(obj["id"], category=category, object=obj)
            # get room id # category|room_id|object_id
            graph.add_edge(f"room|{room_id}", obj["id"])
            if "children" in obj:
                for child in obj["children"]:
                    category = child["id"].split("|")[0]
                    if child["id"] in exclude_ids or category in exclude_categories:
                        continue
                    child["category"] = category
                    graph.add_node(child["id"], category=category, object=child)
                    graph.add_edge(obj["id"], child["id"])
        return graph

    def get_scene_node(self, node_id: int, semantic_info: dict) -> list[SceneNode]:
        if "assetId" in semantic_info:
            obj_shape_info = self.assets_db[semantic_info["assetId"]]
            x, y, z = (
                obj_shape_info["boundingBox"]["x"],
                obj_shape_info["boundingBox"]["y"],
                obj_shape_info["boundingBox"]["z"],
            )
            if "category" in semantic_info:
                category = semantic_info["category"]
                if category in self.default_container_list:
                    shape = ComplexStorage(size=[y, x, z])
                else:
                    shape = Box(size=[x, z, y])
        else:
            shape = Box(size=[0, 0, 0])
        return SceneNode(node_id, shape, semantic_info, is_truth=True)

    def parse_scene_graph(
        self,
        root_node_id: int,
        file_dir: str,
        exclude_ids=None,
        exclude_categories=None,
    ) -> SceneGraph:
        if exclude_categories is None:
            exclude_categories = []
        if exclude_ids is None:
            exclude_ids = []
        exclude_obj_ids = [self.obj2id[id] for id in exclude_ids]
        G = self.parse_graph(exclude_obj_ids, exclude_categories)

        def get_3d_graph():
            node_dict: dict[int, SceneNode] = {}
            edge_dict: dict[int, SceneEdge] = {}

            depth = nx.single_source_shortest_path_length(
                G, source=self.obj2id[root_node_id]
            )
            max_depth = max(depth.values())
            for level in range(max_depth + 1):
                for node in G.nodes:
                    if depth[node] == level:
                        node_id = self.id2obj[node]
                        semantic_info = G.nodes[node]["object"]
                        node_dict[node_id] = self.get_scene_node(node_id, semantic_info)
                        # get parent node
                        parent_nodes = list(G.predecessors(node))
                        if len(parent_nodes) == 0:
                            continue
                        parent_id = self.id2obj[parent_nodes[0]]
                        edge_dict[node_id] = SceneEdge(
                            parent_id, node_id, semantic_info={"relationType": "on"}
                        )
                        # room node
                        if level == 1:
                            # get floorPolygon
                            polygon = G.nodes[node]["object"]["floorPolygon"]
                            x_offsets = []
                            z_offsets = []
                            for obj in polygon:
                                x_offsets.append(obj["x"])
                                z_offsets.append(obj["z"])
                            x_offset = np.mean(x_offsets)
                            z_offset = np.mean(z_offsets)
                            node_dict[node_id].semantic_info["position"] = {
                                "x": x_offset,
                                "y": 0,
                                "z": z_offset,
                            }
                        else:
                            parent_position = node_dict[parent_id].semantic_info[
                                "position"
                            ]
                            child_position = node_dict[node_id].semantic_info[
                                "position"
                            ]
                            x_offset = child_position["x"] - parent_position["x"]
                            z_offset = child_position["z"] - parent_position["z"]

                        edge_dict[node_id].add_relation(
                            node_dict[parent_id],
                            node_dict[node_id],
                            dof_type="x-y",
                            pose=[
                                x_offset * (-1) ** level,
                                z_offset * (-1) ** level,
                                0.0,
                            ],
                        )
            return node_dict, edge_dict, root_node_id

        graph = SceneGraph(
            scene_name="env",
            scene_file=self.house,
            file_dir=file_dir,
            fn=get_3d_graph,
            robot_node_id=root_node_id,
        )
        return graph


def create_asset2info() -> None:
    with open(ASSETS_DB) as f:
        asset_db = json.load(f)
        asset2info = {}
        for _, assets in asset_db.items():
            for asset in assets:
                asset2info[asset["assetId"]] = asset
        with open("asset2info.json", "w") as f:
            json.dump(asset2info, f)


if __name__ == "__main__":
    import prior

    dataset = prior.load_dataset("procthor-10k")["train"]
    p = ProcScene(house=dataset[69], quiet=False)
    graph = p.parse_scene_graph(root_node_id=0, file_dir="data/test")
    p.get_top_down_frame()
