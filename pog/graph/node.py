from enum import Enum

from pog.graph.shape import Shape, ShapeType


class ContainmentState(Enum):
    Closed = 0
    Opened = 1


class Node:
    def __init__(
        self, id: int, shape: Shape, semantic_info=None, motion_info=None, **kwargs
    ) -> None:
        """Class for graph node. The purpose of this class is to assign affordances to current node

        Args:
            id (int): node id
            shape (shape.Shape): node shape
        """
        if semantic_info is None:
            semantic_info = {}
        self.id = id
        self.shape = shape
        self.attributes = kwargs
        self.affordance = shape.get_aff()
        self.accessible = True
        self.is_truth = True
        self.semantic_info = semantic_info
        self.motion_info = motion_info
        if self.shape.object_type == ShapeType.ARTIC:
            self.state = ContainmentState.Closed
        else:
            self.state = None

        for _, value in self.affordance.items():
            value.node_id = self.id

    def __repr__(self) -> str:
        return str(self.shape) + ": " + str(self.id) + f"{self.semantic_info}"

    def __eq__(self, other) -> bool:
        return (
            self.id == other.id and self.state == other.state
        )  # NOTE: we are assuming object has unique id

    @property
    def default_affordance(self):
        return self.affordance[self.shape.default_affordance_name]

    def get_llm_info(self):
        return {
            "id": self.id,
            "is_container": self.shape == ShapeType.ARTIC,
            "is_close": self.state == ContainmentState.Closed,
        }
