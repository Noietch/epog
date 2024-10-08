from array import ArrayType
from typing import Tuple

import numpy as np
# import pybullet as p
import sdf.d2
import sdf.d3
import transforms3d
import trimesh
import trimesh.creation as creation
from pog.graph.params import BULLET_GROUND_OFFSET, WALL_THICKNESS
from pog.graph.shape import Affordance, ShapeID, ShapeType
from pog.graph.shapes.complex_storage import ComplexStorage

# TODO: refactor wardrobe(remove duplicate staffs)
class Wardrobe(ComplexStorage):

    def __init__(self,
                 shape_type=ShapeID.Wardrobe,
                 size=np.array([1.6, 0.8, 2.0]),
                 transform=np.identity(4),
                 storage_type='cabinet',
                 **kwargs):
        """
        size: size in xyz
        """
        super().__init__(shape_type, size, transform, storage_type, **kwargs)

    @property
    def SHAPE_TYPE(self):
        return ShapeID.Wardrobe

    def create_aff(self, storage_type: str, size):
        outer_params = {
            "containment": False,
            "shape": sdf.d2.rectangle(size[[0, 1]]),
            "area": size[0] * size[1],
            "bb": size[[0, 1]],
            "height": size[2],
        }
        inner_params = {
            "containment": True,
            "shape": sdf.d2.rectangle(size[[0, 1]] - 2 * WALL_THICKNESS),
            "area":
            (size[0] - 2 * WALL_THICKNESS) * (size[1] - 2 * WALL_THICKNESS),
            "bb": size[[0, 1]] - 2 * WALL_THICKNESS,
            "height": size[2] - 2 * WALL_THICKNESS,
        }
        if storage_type == 'cabinet':
            aff_dicts = self.get_cabinet_affs(inner_params, outer_params)
            for aff in aff_dicts:
                self.add_aff(
                    Affordance(name=aff["name"],
                               transform=aff["tf"],
                               **aff["params"]))

    def create_bullet_shapes(self, global_transform):
        visual_shapes = []
        collision_shapes = []
        halfwlExtents = [
            self.size[0] / 2., self.size[1] / 2., WALL_THICKNESS / 4.
        ]
        halflhExtents = [
            self.size[0] / 2., WALL_THICKNESS / 4., self.size[2] / 2.
        ]
        halfwhExtents = [
            WALL_THICKNESS / 4., self.size[1] / 2., self.size[2] / 2.
        ]
        door = [self.size[0] / 4., WALL_THICKNESS / 4., self.size[2] / 2.]
        shape_params = [{
            "ext":
            halfwlExtents,
            "frame_position": [0, 0, -self.size[2] / 2. + WALL_THICKNESS / 4.]
        }, {
            "ext": halfwlExtents
        }, {
            "ext": halfwlExtents
        }, {
            "ext": halfwhExtents
        }, {
            "ext": halfwhExtents
        }, {
            "ext": halflhExtents
        }, {
            "ext": door
        }, {
            "ext": door,
        }]
        for param in shape_params:
            visual_shapes.append(
                p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=param["ext"],
                    visualFramePosition=param.get("frame_position", None),
                ))
            collision_shapes.append(
                p.createCollisionShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=param["ext"],
                    collisionFramePosition=param.get("frame_position", None),
                ))

        visual_shapes.append(
            p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.015))
        collision_shapes.append(
            p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.015))

        translation, quaternion = self.parse_transform(global_transform)

        multibody = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shapes[0],
            baseVisualShapeIndex=visual_shapes[0],
            basePosition=translation + BULLET_GROUND_OFFSET,
            baseOrientation=np.hstack((quaternion[1:4], quaternion[0])),
            linkMasses=[0, 0, 0, 0, 0, 1, 0, 0],
            linkCollisionShapeIndices=collision_shapes[1:],
            linkVisualShapeIndices=visual_shapes[1:],
            linkPositions=[
                [0, 0, self.size[2] / 2. - WALL_THICKNESS / 4.],
                [0, -WALL_THICKNESS / 4., 0],
                [-self.size[0] / 2. + WALL_THICKNESS / 4., 0, 0],  # left/right
                [self.size[0] / 2. - WALL_THICKNESS / 4., 0, 0],
                [0, -self.size[1] / 2. + WALL_THICKNESS / 4., 0],  # back board
                [
                    self.size[0] / 4., self.size[1] / 2. + WALL_THICKNESS / 4.,
                    0
                ],
                [
                    -self.size[0] / 4. + WALL_THICKNESS / 4.,
                    self.size[1] / 2. - WALL_THICKNESS / 4., 0
                ],
                [self.size[0] / 4. - 0.1, 0.01, 0],
            ],
            linkOrientations=[
                [0, 0, 0, 1],
                [0, 0, 0, 1],
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
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],
            linkParentIndices=[0, 0, 0, 0, 0, 0, 0, 6],
            linkJointTypes=[
                p.JOINT_FIXED,
                p.JOINT_FIXED,
                p.JOINT_FIXED,
                p.JOINT_FIXED,
                p.JOINT_FIXED,
                p.JOINT_PRISMATIC,
                p.JOINT_FIXED,
                p.JOINT_PRISMATIC,
            ],  # related to door of storage
            linkJointAxis=[
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],  # also joint related
            useMaximalCoordinates=False)
        p.changeDynamics(multibody,
                         5,
                         jointLowerLimit=self.size[1] / 2.,
                         jointUpperLimit=0)
        p.changeVisualShape(multibody,
                            linkIndex=4,
                            rgbaColor=[0.76470588, 0.765, 0.765, 1.],
                            specularColor=[0.4, 0.4, 0])
        return visual_shapes, collision_shapes, multibody
