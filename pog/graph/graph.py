import contextlib
import copy
import json
import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import trimesh
import vedo
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from networkx.drawing.nx_agraph import graphviz_layout

from pog.graph import shape
from pog.graph.edge import Edge
from pog.graph.node import ContainmentState, Node
from pog.graph.params import FRICTION_ANGLE_THRESH
from pog.graph.shapes import ComplexStorage, Cone, Drawer, Wardrobe
from pog.graph.utils import match


def createTestGraph():
    """A test graph

    Returns:
        node_dict (dict: (node_id, node)): A dictionary of nodes
        edge_dict (dict: (child_id, edge)): A dictionary of edges
        root_id (int): Root of test graph
    """
    node_dict = {}
    edge_dict = {}

    node0 = shape.Box(size=[5, 5, 0.2])
    node_node0 = Node(id=0, shape=node0)

    node1 = shape.Sphere()
    node_node1 = Node(id=1, shape=node1)

    node2 = shape.Box(size=[1, 2, 3])
    node_node2 = Node(id=2, shape=node2)

    node3 = shape.Cylinder(height=3, radius=1)
    node_node3 = Node(id=3, shape=node3)

    node4 = shape.Cylinder(height=1, radius=0.5)
    node_node4 = Node(id=4, shape=node4)

    node5 = shape.Box(size=0.6)
    node_node5 = Node(id=5, shape=node5)

    node6 = shape.Box(size=0.9)
    node_node6 = Node(id=6, shape=node6)

    node7 = shape.Cylinder(height=1.5, radius=0.5)
    node_node7 = Node(id=7, shape=node7)

    edge_01 = Edge(parent=0, child=1)
    edge_01.add_relation(
        node_node0.affordance["box_aff_pz"],
        node_node1.affordance["sphere_aff_nz"],
        dof_type="x-y-2dof",
        pose=[1, 1],
    )

    edge_02 = Edge(parent=0, child=2)
    edge_02.add_relation(
        node_node0.affordance["box_aff_pz"],
        node_node2.affordance["box_aff_nx"],
        pose=[-0, -1.2, 0],
    )

    edge_03 = Edge(parent=0, child=3)
    edge_03.add_relation(
        node_node0.affordance["box_aff_pz"],
        node_node3.affordance["cylinder_aff_nz"],
        dof_type="x-y-2dof",
        pose=[-1, 1],
    )

    edge_24 = Edge(parent=2, child=4)
    edge_24.add_relation(
        node_node2.affordance["box_aff_px"],
        node_node4.affordance["cylinder_aff_pz"],
        dof_type="x-y-2dof",
        pose=[-0.4454, -0.4881],
    )

    edge_25 = Edge(parent=2, child=5)
    edge_25.add_relation(
        node_node2.affordance["box_aff_px"],
        node_node5.affordance["box_aff_nx"],
        dof_type="x-y",
        pose=[-0.3679, 0.6258, 0.5991],
    )

    edge_36 = Edge(parent=3, child=6)
    edge_36.add_relation(
        node_node3.affordance["cylinder_aff_pz"],
        node_node6.affordance["box_aff_nx"],
        dof_type="x-y",
        pose=[0, 0, 1],
    )

    edge_27 = Edge(parent=2, child=7)
    edge_27.add_relation(
        node_node2.affordance["box_aff_px"],
        node_node7.affordance["cylinder_aff_nz"],
        dof_type="x-y-2dof",
        pose=[0.3964, 0.1999],
    )

    node_dict[0] = node_node0
    node_dict[1] = node_node1
    node_dict[2] = node_node2
    node_dict[3] = node_node3
    node_dict[4] = node_node4
    node_dict[5] = node_node5
    node_dict[6] = node_node6
    node_dict[7] = node_node7

    edge_dict[1] = edge_01
    edge_dict[2] = edge_02
    edge_dict[3] = edge_03
    edge_dict[4] = edge_24
    edge_dict[5] = edge_25
    edge_dict[6] = edge_36
    edge_dict[7] = edge_27

    root_id = 0

    return node_dict, edge_dict, root_id


class Graph:
    scene: trimesh.Scene

    def __init__(
        self,
        scene_name,
        file_dir=None,
        file_name=None,
        fn=None,
        robot_node_id=None,
        **kwargs,
    ) -> None:
        """class for scene graph and operations on scene graph

        Args:
            scene_name (str): name of scene
            file_path (path, optional): file path of graph. Defaults to None.
            fn (function, optional): function to create graph. Defaults to None.
        """
        self.name = scene_name
        self.graph = nx.DiGraph()
        self.robot = nx.DiGraph()
        self.robot_root = None
        self.node_dict: dict[int, Node]
        self.edge_dict: dict[int, Edge]
        if fn is not None:
            self.node_dict, self.edge_dict, self.root = fn(**kwargs)
        elif file_dir is not None and file_name is not None:
            logging.info(
                f"Loading scene {self.name} from file {os.path.join(file_dir, file_name)}"
            )
            self.node_dict = {}
            self.edge_dict = {}
            with open(os.path.join(file_dir, file_name)) as json_file:
                data = json.load(json_file)
                self.root = data["root"]
                for n in data["nodes"]:
                    if n["shape"] == shape.ShapeID.Sphere.value:
                        temp_node = shape.Sphere.from_saved(n)
                    elif n["shape"] == shape.ShapeID.Box.value:
                        temp_node = shape.Box.from_saved(n)
                    elif n["shape"] == shape.ShapeID.Cylinder.value:
                        temp_node = shape.Cylinder.from_saved(n)
                    elif n["shape"] == shape.ShapeID.Cone.value:
                        temp_node = Cone.from_saved(n)
                    elif n["shape"] == shape.ShapeID.Storage.value:
                        temp_node = shape.Storage.from_saved(n)
                    elif n["shape"] == shape.ShapeID.Imported.value:
                        temp_node = shape.Imported.from_saved(n, file_dir)
                    elif n["shape"] == shape.ShapeID.ComplexStorage.value:
                        temp_node = ComplexStorage.from_saved(n)
                    elif n["shape"] == shape.ShapeID.Wardrobe.value:
                        temp_node = Wardrobe.from_saved(n)
                    elif n["shape"] == shape.ShapeID.Drawer.value:
                        temp_node = Drawer.from_saved(n)
                    else:
                        logging.error(
                            "Unsupported shape type: {}".format(
                                shape.ShapeID(n["shape"])
                            )
                        )

                    self.node_dict[n["id"]] = Node(
                        id=n["id"],
                        shape=temp_node,
                        semantic_info=n["semantic_info"],
                        motion_info=n["motion_info"],
                    )

                for e in data["edges"]:
                    self.edge_dict[e["child"]] = Edge(
                        e["parent"], e["child"], semantic_info=e["semantic_info"]
                    )
                    for relation in e["relations"].values():
                        try:
                            self.edge_dict[e["child"]].add_relation(
                                self.node_dict[e["parent"]].affordance[
                                    relation["parent"]
                                ],
                                self.node_dict[e["child"]].affordance[
                                    relation["child"]
                                ],
                                relation["dof"],
                                relation["pose"],
                            )
                        except KeyError:
                            self.edge_dict[e["child"]].add_relation(
                                self.node_dict[e["parent"]].affordance[
                                    relation["parent"]
                                ],
                                self.node_dict[e["child"]].affordance[
                                    relation["child"]
                                ],
                            )
                if "other_info" in data and data["other_info"]:
                    self.nodes_involved = data["other_info"]["nodes_involved"]
        else:
            logging.error("Requires an initialization method of graph.")

        for _, node in self.node_dict.items():
            self.graph.add_node(node.id, node=node)

        for _, edge in self.edge_dict.items():
            self.graph.add_edge(edge.parent_id, edge.child_id, edge=edge)
            if edge.parent_id == self.root:
                self.root_aff = edge.relations[shape.AffordanceType.Support][
                    "parent"
                ].name

        self.respwan_robot(robot_node_id)
        self.updateCoM()
        self.updateAccessibility()
        self.computeGlobalTF()
        self.createCollisionManager()

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return nx.algorithms.isomorphism.is_isomorphic(
            self.graph, other.graph, node_match=match, edge_match=match
        )

    def add_edge(self, edge: Edge):
        """Add edge to graph

        Args:
            edge (Edge): edge to be added
        """
        self.edge_dict[edge.child_id] = edge
        self.graph.add_edge(edge.parent_id, edge.child_id, edge=edge)

        self.updateCoM()
        self.updateAccessibility()
        self.computeGlobalTF()
        self.createCollisionManager()

    def addEdge(
        self,
        child_id,
        parent_id,
        child_affordance,
        parent_affordance,
        dof_type="x-y",
        pose=None,
        semantic_info=None,
    ):
        if pose is None:
            pose = [0.0, 0.0, 0.0]
        assert parent_id in self.node_dict
        assert child_id in self.node_dict

        self.edge_dict[child_id] = Edge(
            parent=parent_id, child=child_id, semantic_info=semantic_info
        )
        self.edge_dict[child_id].add_relation(
            parent_affordance, child_affordance, dof_type, pose
        )
        self.graph.add_edge(parent_id, child_id, edge=self.edge_dict[child_id])

        self.updateCoM()
        self.updateAccessibility()
        self.computeGlobalTF()
        self.createCollisionManager()

    def removeEdge(self, child_id):
        """remove an edge from edge list

        Args:
            child_id (int): node id of child node of removed edge
        """
        with contextlib.suppress(nx.exception.NetworkXError):
            self.graph.remove_edge(self.edge_dict[child_id].parent_id, child_id)
        del self.edge_dict[child_id]

    def remove_node(self, id):
        """Remove node from graph

        Args:
            id (int): node id
        """
        if not self.graph.has_node(id):
            logging.error(
                f"GraphBase.removeNode(): Cannot find node {id} in environment graph!"
            )

        self.graph.remove_node(id)
        self.node_dict.pop(id)
        self.edge_dict.pop(id)

    def removeNode(self, id, edge=None, ee=None):
        """Remove environment node [id], all its child nodes and their adjacent edges.
        Add removed environment nodes to robot graph and attach to ee.

        Args:
            id (int): node id
            ee (int): end-effector node in robot graph

        """
        if not self.graph.has_node(id):
            logging.error(
                f"GraphBase.removeNode(): Cannot find node {id} in environment graph!"
            )
        if ee is not None and not self.robot.has_node(ee):
            logging.error(
                f"GraphBase.removeNode(): Cannot find end effector {ee} in robot graph!"
            )
        if id == self.root:
            logging.error("GraphBase.removeNode(): Cannot remove root node!")

        self.graph.remove_node(id)
        self.robot.add_node(id, node=self.node_dict[id])
        self.__removeNodeHelper(id)

        self.removeEdge(id)

        if ee is not None:
            assert edge is not None
            self.robot.add_edge(ee, id, edge=edge)
        else:
            self.robot_root = id

        self.updateCoM()
        self.updateAccessibility()
        self.computeGlobalTF()
        self.createCollisionManager()

    def __removeNodeHelper(self, id):
        for child, edge in self.edge_dict.items():
            if edge.parent_id == id and self.graph.has_node(child):
                self.__removeNodeHelper(child)
                self.robot.add_node(child, node=self.node_dict[child])
                self.robot.add_edge(edge.parent_id, child, edge=self.edge_dict[child])
                self.graph.remove_node(child)

    def add_node(self, node: Node):
        """Add external node to scene graph

        Args:
            node (Node): external node to be added
            edge (Edge): edge between external node and existing node in scene graph
        """
        assert node.id not in self.node_dict

        self.node_dict[node.id] = node
        self.graph.add_node(node.id, node=node)

    def add_external_node(self, node: Node):
        """Add external node to scene graph

        Args:
            node (Node): external node to be added
            edge (Edge): edge between external node and existing node in scene graph
        """

        self.node_dict[node.id] = node
        self.graph.add_node(node.id, node=node)

    def addNode(self, parent, edge, object=None):
        """Add node to environment graph, remove node from robot graph

        Args:
            parent (int): The node on environment graph that we move the object to
            edge (Edge): The new edge established between parent and object
            object (int, optional): The node on robot graph that we want to move to environment graph. Defaults to None (Move all objects).
        """
        if not self.graph.has_node(parent):
            logging.error(
                f"GraphBase.addNode(): Cannot find node {parent} in environment graph!"
            )
        if object is not None and not self.robot.has_node(object):
            logging.error(
                f"GraphBase.addNode(): Cannot find node {object} in robot graph!"
            )

        if object is None or object == self.robot_root:
            self.graph = nx.compose(self.graph, self.robot)
            self.robot.clear()
            self.graph.add_edge(parent, self.robot_root, edge=edge)
            self.edge_dict[self.robot_root] = edge
        else:
            self.robot.remove_node(object)
            self.graph.add_node(object, node=self.node_dict[object])
            self.graph.add_edge(parent, object, edge=edge)
            self.__addNodeHelper(id)
            self.removeEdge(object)
            self.edge_dict[self.robot_root] = edge

        self.updateCoM()
        self.updateAccessibility()
        self.computeGlobalTF()
        self.createCollisionManager()

    def __addNodeHelper(self, object):
        for child, parent in self.edge_dict.items():
            if parent == object and self.robot.has_node(child):
                self.__addNodeHelper(child)
                self.graph.add_node(child, node=self.node_dict[child])
                self.graph.add_edge(parent, child, edge=self.edge_dict[child])
                self.robot.remove_node(child)

    def toJson(self, file_dir=None, file_name=None, other_info=None):
        os.makedirs(file_dir, exist_ok=True)
        os.makedirs(file_dir + "/meshes", exist_ok=True)

        nodes = []
        edges = []

        for node_id, node in self.node_dict.items():
            temp_node = {}
            temp_node["id"] = node.id
            temp_node["shape"] = node.shape.shape_type.value
            temp_node["transform"] = node.shape.transform.tolist()
            temp_node["radius"] = getattr(node.shape, "radius", -1.0)
            temp_node["size"] = getattr(node.shape, "size", np.array([])).tolist()
            temp_node["height"] = getattr(node.shape, "height", -1.0)
            if node.shape.shape_type == shape.ShapeID.Imported:
                try:
                    temp_node["file_path"] = node.shape.mesh_dir
                except AttributeError:
                    node.shape.shape.export(
                        file_dir + "/meshes/" + str(node_id) + ".stl"
                    )
                    temp_node["file_path"] = (
                        file_dir + "/meshes/" + str(node_id) + ".stl"
                    )
            temp_node["semantic_info"] = node.semantic_info
            temp_node["motion_info"] = node.motion_info.to_dict()
            nodes.append(temp_node)

        for _, edge in self.edge_dict.items():
            temp_edge = {}
            temp_edge["parent"] = edge.parent_id
            temp_edge["child"] = edge.child_id
            temp_edge["relations"] = {}
            for key, value in edge.relations.items():
                temp_relation = {}
                temp_relation["parent"] = getattr(value["parent"], "name", "")
                temp_relation["child"] = getattr(value["child"], "name", "")
                try:
                    temp_relation["dof"] = value["dof"]
                    temp_relation["pose"] = list(value["pose"])
                except KeyError:
                    pass
                temp_edge["relations"][key.name] = temp_relation
            temp_edge["semantic_info"] = edge.semantic_info
            edges.append(temp_edge)

        with open(os.path.join(file_dir, file_name), "w") as outfile:
            logging.info(
                f"Saving scene {self.name} to file {os.path.join(file_dir, file_name)}"
            )
            json.dump(
                {
                    "nodes": nodes,
                    "edges": edges,
                    "root": self.root,
                    "other_info": other_info,
                },
                outfile,
            )

    def getPose(self, edge_id=None):
        """Get pose of current graph

        Args:
            edge_id (list, optional): A list of edges (child nodes). Defaults to None.

        Returns:
            pose_dict (dict (child, pose)): child node id and its pose
        """
        pose_dict = {}
        if edge_id is None:
            for _, edge in self.edge_dict.items():
                if self.graph.has_node(edge.parent_id) and self.graph.has_node(
                    edge.child_id
                ):
                    pose_dict[edge.child_id] = edge.relations[
                        shape.AffordanceType.Support
                    ]
        else:
            for edge in edge_id:
                if self.graph.has_node(
                    self.edge_dict[edge].parent_id
                ) and self.graph.has_node(self.edge_dict[edge].child_id):
                    pose_dict[self.edge_dict[edge].child_id] = self.edge_dict[
                        edge
                    ].relations[shape.AffordanceType.Support]

        return pose_dict

    def setPose(self, pose):
        """Set pose of scene graph

        Args:
            pose (dict (child, pose)): Child node id and its pose
        """
        for child_id, relation in pose.items():
            if self.graph.has_node(relation["parent"].node_id) and self.graph.has_node(
                relation["child"].node_id
            ):
                self.edge_dict[child_id].add_relation(
                    relation["parent"],
                    relation["child"],
                    relation["dof"],
                    relation["pose"],
                )
                self.graph.add_edge(
                    relation["parent"].node_id,
                    relation["child"].node_id,
                    edge=self.edge_dict[child_id],
                )
        self.updateCoM()
        self.computeGlobalTF()

    def trackDepth(self):
        """Find nodes at each depth and store it in self.depth_dict"""
        max_depth = len(nx.algorithms.dag_longest_path(self.graph))
        node_depth = nx.shortest_path_length(self.graph, self.root)
        self.depth_dict = {}
        for depth in range(0, max_depth):
            temp_depth_list = []
            for key, value in node_depth.items():
                if value == depth:
                    temp_depth_list.append(key)
            self.depth_dict[depth] = temp_depth_list

    def updateAccessibility(self, robot_at_node_id=None):  # For pick and place only
        # TODO: the robot is not at root. Only at root for now
        # if robot_at_node_id is None or robot_at_node_id == self.root:
        #     robot_at_node_id = self.root
        #     self.node_dict[self.root].accessible = True

        max_depth = len(nx.algorithms.dag_longest_path(self.graph))

        for node in self.node_dict.values():
            node.accessible = True

        checked_nodes = []
        for depth in range(1, max_depth):
            node_list_current_depth = self.depth_dict[depth]
            for current_depth_node in node_list_current_depth:
                if (
                    self.edge_dict[current_depth_node].containment
                    and self.node_dict[
                        self.edge_dict[current_depth_node].parent_id
                    ].state
                    == ContainmentState.Closed
                    and current_depth_node not in checked_nodes
                ):
                    sub_tree = dfs_tree(self.graph, current_depth_node)
                    for sub_tree_node in sub_tree.nodes:
                        self.node_dict[sub_tree_node].accessible = False
                        checked_nodes.append(sub_tree_node)

    def updateCoM(self):
        """
        Recursively compute center of mass of current scene and store it in edges.
        CoM stored in each edge is the CoM for all its children.
        """
        self.trackDepth()
        max_depth = len(nx.algorithms.dag_longest_path(self.graph))
        for i in reversed(range(1, max_depth)):
            node_dict = self.depth_dict[i]
            for node_id in node_dict:
                total_mass = self.node_dict[node_id].shape.mass
                self.edge_dict[node_id].relations[shape.AffordanceType.Support][
                    "mass"
                ] = total_mass
                self.edge_dict[node_id].relations[shape.AffordanceType.Support][
                    "com"
                ] = (
                    np.dot(
                        self.edge_dict[node_id].parent_to_child_tf,
                        np.concatenate((self.node_dict[node_id].shape.com, [1])),
                    )[0:3]
                    * total_mass
                )
                for succ in self.graph.successors(node_id):
                    self.edge_dict[node_id].relations[shape.AffordanceType.Support][
                        "mass"
                    ] += self.edge_dict[succ].relations[shape.AffordanceType.Support][
                        "mass"
                    ]
                    total_mass += self.edge_dict[succ].relations[
                        shape.AffordanceType.Support
                    ]["mass"]
                    self.edge_dict[node_id].relations[shape.AffordanceType.Support][
                        "com"
                    ] += (
                        np.dot(
                            self.edge_dict[node_id].parent_to_child_tf,
                            np.concatenate(
                                (
                                    self.edge_dict[succ].relations[
                                        shape.AffordanceType.Support
                                    ]["com"],
                                    [1],
                                )
                            ),
                        )[0:3]
                        * self.edge_dict[succ].relations[shape.AffordanceType.Support][
                            "mass"
                        ]
                    )
                self.edge_dict[node_id].relations[shape.AffordanceType.Support][
                    "com"
                ] /= total_mass

    def show(self):
        """Show scene graph and robot graph"""
        plt.figure()
        node_labels = {}
        for key in self.node_dict:
            if self.graph.has_node(key):
                node_labels[key] = str(self.node_dict[key])
        pos = graphviz_layout(self.graph, prog="dot")
        nx.draw_networkx_nodes(
            self.graph, pos, cmap=plt.get_cmap("jet"), node_size=1500
        )
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_weight="bold")
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=self.graph.edges,
            edge_color="r",
            arrows=True,
            width=2,
            arrowsize=50,
        )
        plt.title("Environment Model")

        plt.figure()
        node_labels = {}
        for key in self.node_dict:
            if self.robot.has_node(key):
                node_labels[key] = str(self.node_dict[key])
        pos = graphviz_layout(self.robot, prog="dot")
        nx.draw_networkx_nodes(
            self.robot, pos, cmap=plt.get_cmap("jet"), node_size=1500
        )
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_weight="bold")
        nx.draw_networkx_edges(
            self.robot,
            pos,
            edgelist=self.robot.edges,
            edge_color="r",
            arrows=True,
            width=2,
            arrowsize=50,
        )
        plt.title("Robot Model")

        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")
        plt.close()

    def create_scene(self):
        """create Trimesh.Scene for visualization"""
        max_depth = len(nx.algorithms.dag_longest_path(self.graph))
        node_depth = nx.shortest_path_length(self.graph, self.root)
        geom = self.node_dict[self.root].shape.shape.copy()
        self.scene = trimesh.Scene()
        self.scene.add_geometry(geom, node_name=self.root)
        for i in range(1, max_depth):
            for key, value in node_depth.items():
                if value == i:
                    self.scene.add_geometry(
                        self.node_dict[key].shape.shape.copy(),
                        node_name=key,
                        parent_node_name=self.edge_dict[key].parent_id,
                        transform=self.edge_dict[key].parent_to_child_tf,
                    )

    def genMesh(self, outfile="out.stl"):
        """create mesh of scene graph and save it to a directory

        Args:
            outfile (str, optional): file path. Defaults to 'out.stl'.
        """
        self.create_scene()
        self.scene.dump(concatenate=True).export(outfile)

    def computeGlobalTF(self):
        """Compute transformations from root to all nodes in scene graph"""
        max_depth = len(nx.algorithms.dag_longest_path(self.graph))
        node_depth = nx.shortest_path_length(self.graph, self.root)
        self.global_transform = {}
        self.global_transform[self.root] = np.identity(4)
        for i in range(1, max_depth):
            for key, value in node_depth.items():
                if value == i:
                    tf = np.array(
                        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
                    )
                    tf = self.__genGlobalTFHelper(key, value, tf)
                    try:
                        self.global_transform[key] = tf
                    except KeyError:
                        logging.error(
                            f"Cannot find edge of child ID {key} in edge list."
                        )

    def __genGlobalTFHelper(self, key, value, tf):
        if value == 0:
            return tf
        elif value > 0:
            value -= 1
            tf = np.dot(self.edge_dict[key].parent_to_child_tf, tf)
            return self.__genGlobalTFHelper(self.edge_dict[key].parent_id, value, tf)

    def getSubGraph(self, node_id=None, depth=None, includes_nodes=None):
        """Get subgraph given a node in current scene graph. Return a deepcopy of self if node_id is None

        Args:
            node_id (int): node id
            depth (int, optional): The depth of subgraph. Defaults to None.

        Returns:
            graph (Graph): The subgraph of scene graph with root at nood_id
        """
        if node_id is None:
            node_id = self.root
        else:
            assert self.graph.has_node(node_id)
        sg = dfs_tree(self.graph, node_id, depth_limit=depth)

        def fn():
            node_dict = {}
            edge_dict = {}

            for item in sg.nodes:
                if includes_nodes:
                    if item in includes_nodes:
                        node_dict[item] = copy.deepcopy(self.node_dict[item])
                else:
                    node_dict[item] = copy.deepcopy(self.node_dict[item])

            for item in sg.edges:
                if includes_nodes:
                    if item[1] in includes_nodes:
                        if item[0] in includes_nodes:
                            edge_dict[item[1]] = copy.deepcopy(self.edge_dict[item[1]])
                        else:
                            logging.debug("debug")
                else:
                    edge_dict[item[1]] = copy.deepcopy(self.edge_dict[item[1]])

            root_id = node_id
            return node_dict, edge_dict, root_id

        sub_graph = Graph("subgraph", fn=fn, robot_node_id=self.robot_node_id)
        return sub_graph

    def copy(self):
        def fn():
            node_dict = {}
            edge_dict = {}
            for item in self.graph.nodes:
                node_dict[item] = copy.deepcopy(self.node_dict[item])
            for item in self.graph.edges:
                if item != self.root:
                    edge_dict[item[1]] = copy.deepcopy(self.edge_dict[item[1]])
            root_id = self.root
            return node_dict, edge_dict, root_id

        graph_copy = Graph("graph copy", fn=fn)
        graph_copy.root_aff = self.root_aff
        graph_copy.robot_node_id = self.robot_node_id

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

    def createCollisionManager(self):
        """Create Trimesh.collision.CollisionManager for current scene"""
        self.collision_manager = trimesh.collision.CollisionManager()
        for key, value in self.node_dict.items():
            if key in self.graph.nodes():
                try:
                    self.collision_manager.add_object(
                        name=str(key),
                        mesh=value.shape.shape,
                        transform=self.global_transform[key],
                    )
                except (KeyError, AttributeError):
                    continue
                    # print(key, self.global_transform.keys())
                    # self.show()

    def checkStability(self):
        """Check if self is stable

        Returns:
            (bool): True if self is stable
        """
        stable = True
        unstable_node = []
        self.computeGlobalTF()
        vertical_dir = self.node_dict[self.root].affordance[self.root_aff].get_axes()
        for node_id in self.edge_dict:
            if node_id not in self.graph.nodes:
                continue
            parent_id = self.edge_dict[node_id].parent_id
            parent_aff_name = (
                self.edge_dict[node_id]
                .relations[shape.AffordanceType.Support]["parent"]
                .name
            )
            tf = (
                self.global_transform[parent_id]
                @ self.node_dict[parent_id].affordance[parent_aff_name].transform
            )
            uv1 = tf[0:3, 2] / np.linalg.norm(tf[0:3, 2])
            uv2 = vertical_dir / np.linalg.norm(vertical_dir)
            angle = np.arccos(np.dot(uv1, uv2))
            if angle > FRICTION_ANGLE_THRESH:
                stable = False
                unstable_node.append(node_id)
        return (stable, unstable_node)

    def respwan_robot(self, robot_node_id: int):
        self.robot_node_id = robot_node_id

    def get_distance_btw_nodes(self, node1_id: int, node2_id: int):
        node1_transform = self.global_transform[node1_id]
        node2_transform = self.global_transform[node2_id]
        x1 = node1_transform[:, -1][:3]
        x2 = node2_transform[:, -1][:3]
        return np.linalg.norm(x1 - x2)

    def viuslize_graph(self):
        ploter = vedo.Plotter()
        self.create_scene()
        ploter.add(self.scene.dump(concatenate=True))
        # for node in self.node_dict.values():
        #     if 1 <= node.semantic_info['level']:
        #         transform = self.global_transform[node.id]
        #         x, y, z = transform[:, -1][:3]
        #         text = vedo.Text3D(f"{node.semantic_info['class_name']}-{int(node.id)}",
        #                            pos=(x, y, 4), s=0.1, c='black')
        #         ploter.add(text)
        ploter.show(axes=1).close()

    def get_subgraph(self, root_id: int):
        return nx.descendants(self.graph, root_id)
