import json
import os
import random

import networkx as nx
import prior
from loguru import logger
from tqdm import tqdm

from epog.algorithm.epog.fake_simulator import MotionErrorType
from epog.data_gen.task_list import task_list as task_list_defined
from epog.envs.proc_scene import ProcScene, SceneEdge, SceneGraph, SceneNode
from pog.graph.shape import Box


class ProcTask:
    task_list = task_list_defined
    defualt_floor_plan = {"bedroom": 1, "kitchen": 1, "livingroom": 1, "bathroom": 1}
    max_exception = 2

    def __init__(self) -> None:
        self.dataset = prior.load_dataset("procthor-10k")["train"]
        self.seed = random.seed(42)

    def check_task_included(self, scene: dict, task: str = None):
        scene: ProcScene = ProcScene(scene, quiet=True)
        graph = scene.parse_graph()  # use the normal graph to speed up the process
        task_info = self.task_list[task]
        scene_flag = True
        include_objects = {}
        for room in task_info.room_includes:
            room_flag = False
            for node_id in graph.nodes:
                node = graph.nodes[node_id]
                if node["category"].lower() == room.lower():
                    child_nodes_id = nx.descendants(graph, node_id)
                    object_required = set(task_info.filter_object_init_room[room])
                    child_node_categories = set()
                    for child_node_id in child_nodes_id:
                        node = graph.nodes[child_node_id]
                        child_node_categories.add(node["category"])
                    if object_required.issubset(child_node_categories):
                        for object_category in object_required:
                            for child_node_id in child_nodes_id:
                                node = graph.nodes[child_node_id]
                                if node["category"] == object_category:
                                    include_objects[object_category] = scene.id2obj[
                                        child_node_id
                                    ]
                        room_flag = True
                        break
            scene_flag &= room_flag
        include_relations = []
        for relation in task_info.relations:
            if (
                relation.children in include_objects
                and relation.parent in include_objects
            ):
                include_relations.append(
                    [
                        include_objects[relation.children],
                        relation.relationType,
                        include_objects[relation.parent],
                    ]
                )
        return scene_flag, include_relations

    def check_scene_floor_plan(self, scene: dict):
        floor_plan = {"bedroom": 0, "kitchen": 0, "livingroom": 0, "bathroom": 0}
        for room in scene["rooms"]:
            floor_plan[room["roomType"].lower()] += 1
        # check if the floor plan is valid
        for room, num in floor_plan.items():
            if num != self.defualt_floor_plan[room]:
                return False
        return True

    def filter_scene(self, task: str, num_scene: int = 10):
        scene_count = 0
        filtered_scene = []
        logger.info(f"Filtering scene for task {task}")
        with tqdm(total=num_scene) as pbar:
            for i in range(len(self.dataset)):
                scene = self.dataset[i]
                if self.check_scene_floor_plan(scene):
                    scene_flag, include_objects = self.check_task_included(scene, task)
                    if scene_flag:
                        filtered_scene.append(
                            {"scene_id": i, "include_relations": include_objects}
                        )
                        scene_count += 1
                        pbar.update(1)
                if scene_count >= num_scene:
                    break
        return filtered_scene

    def get_task_graph(
        self, goal_graph: SceneGraph, task_list: list[dict]
    ) -> SceneGraph:
        # add all the nodes that are involved in the task
        node_involved = set()
        for task in task_list:
            ancestor = goal_graph.get_ancestors(task["children"])
            for node_id in ancestor:
                node_involved.add(node_id)

        # add room node and receptacle node
        node_bases = goal_graph.get_base_nodes_id()
        for node_id in node_bases:
            node_involved.add(node_id)

        return goal_graph.copy(node_involved)

    def sample_exceptions(
        self, task_name: str, task_list: list[list], start_graph: SceneGraph
    ):
        exception_types = self.task_list[task_name].exceptions
        exception_cnt = 0
        exception_nodes = set()
        for relation in task_list:
            if exception_cnt < self.max_exception:
                exception_type = random.choice(exception_types)
                exception_cnt += 1
                # create a virtual node
                semantic_info = {"category": "Blocks"}
                shape = Box(size=[0, 0, 0])
                new_node_id = max(start_graph.node_dict.keys()) + 1
                new_node = SceneNode(new_node_id, shape, semantic_info, is_truth=True)
                start_graph.add_external_node(new_node)
                # add relation
                if exception_type == MotionErrorType.CollisionError:
                    _, _, parent_id = relation
                    node_parent = start_graph.node_dict[parent_id]
                    node_parent.motion_info.collision_nodes_id.append(new_node_id)
                    exception_nodes.add(parent_id)
                elif exception_type == MotionErrorType.StabilityError:
                    child_id, _, _ = relation
                    node_child = start_graph.node_dict[child_id]
                    node_child.motion_info.unstable_nodes_id.append(new_node_id)
                    exception_nodes.add(child_id)
                elif exception_type == MotionErrorType.BlockError:
                    child_id, _, _ = relation
                    node_child = start_graph.node_dict[child_id]
                    node_child.motion_info.block_nodes_id.append(new_node_id)
                    exception_nodes.add(child_id)
                else:
                    raise ValueError(f"Invalid exception type {exception_type}")
        return list(exception_nodes)

    def gen_task_by_id(
        self,
        scene_id: int,
        task_list: dict,
        task_name: str,
        save_path: str = "data/task",
    ):
        scene = self.dataset[scene_id]
        # parse scene
        proc_scene = ProcScene(scene, quiet=True)
        start_graph: SceneGraph = proc_scene.parse_scene_graph(
            root_node_id=0, file_dir=save_path
        )
        # exception sampling
        exception_nodes = self.sample_exceptions(task_name, task_list, start_graph)
        # get goal graph
        goal_graph = start_graph.copy()
        pog_task_list = []
        for relation in task_list:
            children_id, relation_type, parent_id = relation
            node_child = goal_graph.node_dict[children_id]
            node_parent = goal_graph.node_dict[parent_id]
            pog_task_list.append(
                {
                    "children": children_id,
                    "relationType": relation_type,
                    "parent": parent_id,
                }
            )
            goal_graph.removeEdge(children_id)
            new_edge = SceneEdge(
                parent_id, children_id, semantic_info={"relationType": relation_type}
            )
            new_edge.add_relation(node_parent, node_child)
            goal_graph.add_edge(new_edge)
        # save task
        save_path = os.path.join(save_path, f"scene_{scene_id}")
        task_graph: SceneGraph = self.get_task_graph(goal_graph, pog_task_list)
        goal_graph.toJson(file_dir=save_path, file_name=f"{scene_id}_goal.json")
        start_graph.toJson(file_dir=save_path, file_name=f"{scene_id}_start.json")
        task_graph.toJson(file_dir=save_path, file_name=f"{scene_id}_task.json")
        task_json = {
            "scene_id": scene_id,
            "task_name": task_name,
            "include_room": self.task_list[task_name].room_includes,
            "exception_nodes": exception_nodes,
        }
        with open(os.path.join(save_path, f"{scene_id}_task_info.json"), "w") as f:
            f.write(json.dumps(task_json))
        with open(os.path.join(save_path, f"{scene_id}_scene.json"), "w") as f:
            f.write(json.dumps(scene))
        goal_graph.save_gexf(save_path=os.path.join(save_path, f"{scene_id}_goal.gexf"))
        start_graph.save_gexf(
            save_path=os.path.join(save_path, f"{scene_id}_start.gexf")
        )
        task_graph.save_gexf(save_path=os.path.join(save_path, f"{scene_id}_task.gexf"))

    def gen_task_by_name(
        self, task_name: str, num_scene: int = 10, save_path: str = "data/task"
    ):
        save_path_extends = os.path.join(save_path, task_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filtered_dataset = self.filter_scene(task=task_name, num_scene=num_scene)
        scene_id_list = []
        for filtered_scene in filtered_dataset:
            if filtered_scene["scene_id"] == 4752:  # skip the scene that has error
                continue
            self.gen_task_by_id(
                filtered_scene["scene_id"],
                filtered_scene["include_relations"],
                task_name,
                save_path_extends,
            )
            scene_id_list.append(filtered_scene["scene_id"])
        with open(os.path.join(save_path_extends, "scene_list.json"), "w") as f:
            f.write(json.dumps(scene_id_list))

    def gen_task(self, num_scene: int = 10, save_path: str = "data/task"):
        for task_name in self.task_list:
            self.gen_task_by_name(task_name, num_scene=num_scene, save_path=save_path)


if __name__ == "__main__":
    proc_task = ProcTask()
    proc_task.gen_task(num_scene=10)
