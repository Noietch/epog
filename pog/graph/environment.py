# For visualization and simulation in physics engine
import contextlib
import logging
import os
import sys
import time

import numpy as np
import pybullet as p
import pybullet_data
import transforms3d as tf3d
import vedo
from PIL import Image

from pog.graph.graph import Graph
from pog.graph.params import (
    BULLET_GROUND_OFFSET,
    COLOR,
    COLOR_DICT,
    GEN3_7DOF_POSES,
    WALL_THICKNESS,
)
from pog.graph.shape import AffordanceType, ShapeID


# Context manager to suppress pybullet output
@contextlib.contextmanager
def HideOutput():
    devnull = open(os.devnull, "w")  # noqa: SIM115
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()


class Environment:
    def __init__(
        self,
        graph: Graph,
        display=True,
        options="",
        *,
        input_required=False,
        keep_env=False,
        export_image_path="./result/test_image.png",
    ) -> None:
        with HideOutput():
            if display:
                self.physicsClient = p.connect(p.GUI, options=options)
            else:
                self.physicsClient = p.connect(p.DIRECT, options=options)
            # try:
            #     renderer = P3dRenderer(multisamples=4)
            #     plugin = RenderingPlugin(self.physicsClient, renderer)
            # except Exception as e:
            #     logging.info('apply renderer failed')

            # self.camera_params = {
            #     'y': 30,
            #     'p': -50,
            #     'r': 0,
            #     'dist': 5.,
            #     'target_pos': [0.3, -0.3, 0]
            # }
            self.camera_params = {
                "y": 0,
                "p": -89.9,
                "r": 0,
                "dist": 2.0,
                "target_pos": [0.1, -1.0, 0],
            }

            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setPhysicsEngineParameter(
                numSolverIterations=100,
                numSubSteps=10,
                solverResidualThreshold=1e-9,
                restitutionVelocityThreshold=1e-3,
            )
            p.setTimeStep(1.0 / 240.0)
            p.loadURDF("plane100.urdf", useMaximalCoordinates=True)

            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        self.graph = graph

        self.visualShapeId = {}
        self.collisionShapeId = {}
        self.multibody_dict = {}

        for edge in self.graph.edge_dict.values():
            if edge.parent_id == self.graph.root:
                BULLET_GROUND_OFFSET[-1] += (
                    edge.relations[AffordanceType.Support]["parent"].attributes[
                        "height"
                    ]
                    / 2.0
                )
                break

        # N_storage = 0
        # idx = [3, 5, 6]
        for node_id, node in self.graph.node_dict.items():
            # if node_id not in idx: continue
            shape_type = node.shape.shape_type
            if shape_type == ShapeID.Box:
                self.__create_bullet_shape(
                    node_id, shape_type=p.GEOM_BOX, halfExtents=node.shape.size / 2.0
                )
            elif shape_type == ShapeID.Cylinder:
                self.visualShapeId[node_id] = p.createVisualShape(
                    shapeType=p.GEOM_CYLINDER,
                    radius=node.shape.radius,
                    length=node.shape.height,
                )  # length is different from height
                self.collisionShapeId[node_id] = p.createCollisionShape(
                    shapeType=p.GEOM_CYLINDER,
                    radius=node.shape.radius,
                    height=node.shape.height,
                )
            elif shape_type == ShapeID.Sphere:
                self.__create_bullet_shape(
                    node_id, p.GEOM_SPHERE, radius=node.shape.radius
                )
            elif shape_type == ShapeID.Imported:
                self.__create_bullet_shape(
                    node_id,
                    p.GEOM_MESH,
                    fileName=node.shape.mesh_dir.replace(".stl", ".obj"),
                )
                # self.visualShapeId[node_id] = p.createVisualShape(
                #     shapeType=p.GEOM_MESH,
                #     fileName=node.shape.mesh_dir.replace('.stl', '.obj'))
                # self.collisionShapeId[node_id] = p.createCollisionShape(
                #     shapeType=p.GEOM_MESH,
                #     fileName=node.shape.mesh_dir.replace('.stl', '.obj'))
            elif shape_type == ShapeID.Storage:
                self.create_storage(node_id, node)
                continue
            elif shape_type in [ShapeID.Cone, ShapeID.OpenShelf]:
                self.__create_bullet_shape(
                    node_id,
                    shape_type=p.GEOM_MESH,
                    fileName=node.shape.export_file_name,
                )
            elif shape_type in [
                ShapeID.Wardrobe,
                ShapeID.ComplexStorage,
                ShapeID.Drawer,
            ]:
                visual_shapes, collision_shapes, multibody = (
                    node.shape.create_bullet_shapes(
                        self.graph.global_transform[node_id]
                    )
                )
                self.visualShapeId[node_id] = visual_shapes
                self.collisionShapeId[node_id] = collision_shapes
                self.multibody_dict[node_id] = multibody
                self.colorize(node_id)
                continue
            else:
                logging.warning(f"Unsupported shape type: {node.shape.shape_type}.")

            if node_id == self.graph.root:
                self.multibody_dict[node_id] = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=self.collisionShapeId[node_id],
                    baseVisualShapeIndex=self.visualShapeId[node_id],
                    basePosition=BULLET_GROUND_OFFSET,
                    useMaximalCoordinates=False,
                )
            else:
                translation, rotation, _, _ = tf3d.affines.decompose44(
                    self.graph.global_transform[node_id]
                )
                quaternion = tf3d.quaternions.mat2quat(rotation)
                if translation[-1] < 0:
                    rotx180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                    translation = rotx180 @ translation
                    rotation = rotx180 @ rotation
                    quaternion = tf3d.quaternions.mat2quat(rotation)
                if (
                    self.graph.edge_dict[node_id].relations[AffordanceType.Support][
                        "dof"
                    ]
                    == "fixed"
                ):
                    self.multibody_dict[node_id] = p.createMultiBody(
                        baseMass=0,  # static object
                        baseCollisionShapeIndex=self.collisionShapeId[node_id],
                        baseVisualShapeIndex=self.visualShapeId[node_id],
                        basePosition=translation + BULLET_GROUND_OFFSET,
                        baseOrientation=np.hstack((quaternion[1:4], quaternion[0])),
                        useMaximalCoordinates=False,
                    )
                else:
                    # if node_id == 5:
                    #     self.multibody_dict[node_id] = p.createMultiBody(baseMass=node.shape.mass,
                    #                     baseCollisionShapeIndex=self.collisionShapeId[node_id],
                    #                     baseVisualShapeIndex=self.visualShapeId[node_id],
                    #                     # basePosition=np.array([0,0.07,1.15]) + BULLET_GROUND_OFFSET,
                    #                     # baseOrientation = [0.5,0.5,-0.5,0.5],
                    #                     basePosition=translation + BULLET_GROUND_OFFSET,
                    #                     baseOrientation = [0.7071,0.0,-0.0,0.7071],
                    #                     useMaximalCoordinates=False)
                    # else:
                    self.multibody_dict[node_id] = p.createMultiBody(
                        baseMass=node.shape.mass,
                        baseCollisionShapeIndex=self.collisionShapeId[node_id],
                        baseVisualShapeIndex=self.visualShapeId[node_id],
                        basePosition=translation + BULLET_GROUND_OFFSET,
                        baseOrientation=np.hstack((quaternion[1:4], quaternion[0])),
                        useMaximalCoordinates=False,
                    )
            self.colorize(node_id)

        # p.resetDebugVisualizerCamera(cameraPitch=-50, cameraYaw=45, cameraDistance=0.3, cameraTargetPosition=[-0.05,0.05,0.0])

        # width, height, view_matrix, projection_matrix = self.get_debug_visualizer()[:4]
        if export_image_path is not None:
            self.__get_image(export_image_path)
        if display:
            self.show()
            # For recording
            # while (step < 240*1):
            #     step += 1
            #     if step > 60:
            #         p.stepSimulation()
            #     time.sleep(1./60.)
            if not keep_env:
                self.start_loop(None, input_required)
        if not keep_env:
            p.disconnect(self.physicsClient)

    def colorize(self, node_id: int):
        for link in range(-1, p.getNumJoints(self.multibody_dict[node_id])):
            p.changeVisualShape(
                self.multibody_dict[node_id],
                linkIndex=link,
                rgbaColor=COLOR[COLOR_DICT[node_id]],
                specularColor=[0.4, 0.4, 0],
            )

    def show(self):
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)
        p.resetDebugVisualizerCamera(
            cameraPitch=self.camera_params["p"],
            cameraYaw=self.camera_params["y"],
            cameraDistance=self.camera_params["dist"],
            cameraTargetPosition=self.camera_params["target_pos"],
        )
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def start_loop(self, loop_fn=None, input_required=False):
        while 1:
            # print(p.getJointStates(gen3, [0,1,2,3,4,5,6]))
            if input_required:
                text = input("input q to continue, any key to keep simulating\n\r")
                if text == "q":
                    break
            if loop_fn is not None:
                loop_fn()
            p.stepSimulation()
            time.sleep(1.0 / 60.0)

    def load_ur5(self):
        self._load_robot(
            "pog/graph/robot_description/urdf/deploy_ur5e_base.urdf", [0.3, -1.9, 0.2]
        )

    def load_kinova(self):
        self._load_robot(
            "pog/graph/robot_description/kortex_description/gen3_robotiq_2f_85.urdf",
            [0.5, -0.9, 0.7],
            GEN3_7DOF_POSES["open"],
        )

    def _load_robot(self, urdf_file, start_pose, start_joint_pose=None):
        if start_joint_pose is None:
            start_joint_pose = []
        with HideOutput():
            self.robot = p.loadURDF(
                urdf_file, start_pose, [0, 0, 0, 1], useFixedBase=True
            )
            for joint_id in range(p.getNumJoints(self.robot)):
                try:
                    p.resetJointState(self.robot, joint_id, start_joint_pose[joint_id])
                except IndexError:
                    p.resetJointState(self.robot, joint_id, 0.0)

    def __create_bullet_shape(self, node_id, shape_type, **kwargs):
        self.visualShapeId[node_id] = p.createVisualShape(
            shapeType=shape_type, **kwargs
        )
        self.collisionShapeId[node_id] = p.createCollisionShape(
            shapeType=shape_type, **kwargs
        )

    def __get_image(self, export_image_path):
        width = 640
        height = 640
        p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_params[
                "target_pos"
            ],  # center [0.3, -0.3, 0.0] @ y45, b45
            yaw=self.camera_params["y"],
            pitch=self.camera_params["p"],
            roll=self.camera_params["r"],
            distance=self.camera_params["dist"],
            upAxisIndex=2,
        )
        p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width / height),
            # aspect=1,
            nearVal=0.1,
            farVal=100.0,
        )
        _, _, px, _, _ = p.getCameraImage(
            width=width,
            height=height,
            # viewMatrix=view_matrix,
            # projectionMatrix=proj_matrix,
            # flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            # renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.physicsClient,
        )
        rgba = np.array(px)
        # rgb_array = rgba[:, :, :4]
        # print(np.amin(rgba[3]))
        im = Image.fromarray(rgba, "RGBA")
        im.save(export_image_path, format="PNG")
        # with Image.open(export_image_path) as img:
        #     print(np.amin(np.array(img)))

    def create_storage(self, node_id, node):
        halfwlExtents = [
            node.shape.size[0] / 2.0,
            node.shape.size[1] / 2.0,
            WALL_THICKNESS / 4.0,
        ]
        halfwhExtents = [
            node.shape.size[0] / 2.0,
            WALL_THICKNESS / 4.0,
            node.shape.size[2] / 2.0,
        ]
        halflhExtents = [
            WALL_THICKNESS / 4.0,
            node.shape.size[1] / 2.0,
            node.shape.size[2] / 2.0,
        ]

        self.visualShapeId[node_id] = []
        self.collisionShapeId[node_id] = []

        self.visualShapeId[node_id].append(
            p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=halfwlExtents,
                visualFramePosition=[
                    0,
                    0,
                    -node.shape.size[2] / 2.0 + WALL_THICKNESS / 4.0,
                ],
            )
        )
        self.collisionShapeId[node_id].append(
            p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=halfwlExtents,
                collisionFramePosition=[
                    0,
                    0,
                    -node.shape.size[2] / 2.0 + WALL_THICKNESS / 4.0,
                ],
            )
        )
        self.visualShapeId[node_id].append(
            p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=halfwlExtents)
        )
        self.collisionShapeId[node_id].append(
            p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=halfwlExtents)
        )

        self.visualShapeId[node_id].append(
            p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=halfwhExtents)
        )
        self.collisionShapeId[node_id].append(
            p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=halfwhExtents)
        )
        self.visualShapeId[node_id].append(
            p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=halfwhExtents)
        )
        self.collisionShapeId[node_id].append(
            p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=halfwhExtents)
        )

        self.visualShapeId[node_id].append(
            p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=halflhExtents)
        )
        self.collisionShapeId[node_id].append(
            p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=halflhExtents)
        )
        self.visualShapeId[node_id].append(
            p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=halflhExtents,
                visualFramePosition=[
                    0,
                    -node.shape.size[1] / 2.0 + WALL_THICKNESS / 4.0,
                    0,
                ],
            )
        )
        self.collisionShapeId[node_id].append(
            p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=halflhExtents,
                collisionFramePosition=[
                    0,
                    -node.shape.size[1] / 2.0 + WALL_THICKNESS / 4.0,
                    0,
                ],
            )
        )

        self.visualShapeId[node_id].append(
            p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.015)
        )
        self.collisionShapeId[node_id].append(
            p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.015)
        )

        translation, rotation, _, _ = tf3d.affines.decompose44(
            self.graph.global_transform[node_id]
        )
        quaternion = tf3d.quaternions.mat2quat(rotation)
        if translation[-1] < 0:
            rotx180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            translation = rotx180 @ translation
            rotation = rotx180 @ rotation
            quaternion = tf3d.quaternions.mat2quat(rotation)
        self.multibody_dict[node_id] = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=self.collisionShapeId[node_id][0],
            baseVisualShapeIndex=self.visualShapeId[node_id][0],
            basePosition=translation + BULLET_GROUND_OFFSET,
            baseOrientation=np.hstack((quaternion[1:4], quaternion[0])),
            linkMasses=[0, 0, 0, 0, 1, 0],
            linkCollisionShapeIndices=self.collisionShapeId[node_id][1:],
            linkVisualShapeIndices=self.visualShapeId[node_id][1:],
            linkPositions=[
                [0, 0, node.shape.size[2] / 2.0 - WALL_THICKNESS / 4.0],
                [0, -node.shape.size[1] / 2.0 + WALL_THICKNESS / 4.0, 0],
                [0, node.shape.size[1] / 2.0 - WALL_THICKNESS / 4.0, 0],
                [node.shape.size[0] / 2.0 - WALL_THICKNESS / 4.0, 0, 0],
                [
                    -node.shape.size[0] / 2.0 - WALL_THICKNESS / 4.0,
                    node.shape.size[1] / 2.0 - WALL_THICKNESS / 4.0,
                    0,
                ],
                [-0.01, -node.shape.size[1] + 0.1, 0],
            ],
            linkOrientations=[
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            linkInertialFramePositions=[
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            linkInertialFrameOrientations=[
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            linkParentIndices=[0, 0, 0, 0, 0, 5],
            linkJointTypes=[
                p.JOINT_FIXED,
                p.JOINT_FIXED,
                p.JOINT_FIXED,
                p.JOINT_FIXED,
                p.JOINT_REVOLUTE,
                p.JOINT_FIXED,
            ],  # related to door of storage
            linkJointAxis=[
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 0],
            ],  # also joint related
            useMaximalCoordinates=False,
        )

        p.changeDynamics(
            self.multibody_dict[node_id], 4, jointLowerLimit=-3.14, jointUpperLimit=0
        )
        p.changeVisualShape(
            self.multibody_dict[node_id],
            linkIndex=4,
            rgbaColor=[0.76470588, 0.76470588, 0.76470588, 1.0],
            specularColor=[0.4, 0.4, 0],
        )

    def get_link_tf(self, target_id, target_link, source_id=None, source_link=None):
        target_link_state = p.getLinkState(
            target_id,
            target_link,
            computeLinkVelocity=False,
            computeForwardKinematics=True,
        )
        target_p = target_link_state[4]
        target_q = target_link_state[5]
        try:
            source_link_state = p.getLinkState(
                source_id,
                source_link,
                computeLinkVelocity=False,
                computeForwardKinematics=True,
            )
            source_p = source_link_state[4]
            source_q = source_link_state[5]
            source_quat = source_q[-1]
            source_quat.append(source_q[0:3])
            source_tf = tf3d.affines.compose(
                source_p, tf3d.quaternions.quat2mat(source_quat), [1.0, 1.0, 1.0]
            )
            target_quat = target_q[-1]
            target_quat.append(target_q[0:3])
            target_tf = tf3d.affines.compose(
                target_p, tf3d.quaternions.quat2mat(target_quat), [1.0, 1.0, 1.0]
            )
            source2target_tf = target_tf @ np.linalg.inv(source_tf)
            translation, orientation, _, _ = tf3d.affines.decompose44(source2target_tf)
            orientation = tf3d.quaternions.mat2quat(orientation)
            orientation = np.array([orientation[1:], orientation[0]])
        except (ValueError, np.linalg.LinAlgError):
            translation = target_p
            orientation = target_q
        # temp_body = p.createMultiBody(
        #                 baseMass=0,
        #                 baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.015),
        #                 basePosition=target_link_state[4],
        #                 baseOrientation=target_link_state[5],
        #                 useMaximalCoordinates=False)
        # print(target_link_state)
        return (translation, orientation)


if __name__ == "__main__":
    for i in range(17):
        g = Graph(
            "Test Scene",
            file_dir="pog_example/iros_2022_exp/exp2/result/",
            file_name=f"{i}.json",
        )
        # env = Environment(g)
        g.create_scene()
        vedo.show(g.scene.dump(concatenate=True), axes=0)
