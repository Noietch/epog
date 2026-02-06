from dataclasses import dataclass

from epog.algorithm.epog.fake_simulator import MotionErrorType


@dataclass
class Relationship:
    children: str
    relationType: str
    parent: str


@dataclass
class HouseholdTask:
    task_name: str
    room_includes: list[str]
    filter_object_init_room: dict[str, list[str]]
    relations: list[Relationship]
    exceptions: list[MotionErrorType]


task_list = {
    "prepare_breakfast": HouseholdTask(
        task_name="prepare_breakfast",
        room_includes=["Kitchen", "Livingroom"],
        filter_object_init_room={
            "Kitchen": ["Apple", "Bread", "Plate", "Fork"],
            "Livingroom": ["DiningTable"],
        },
        relations=[
            Relationship("Apple", "on", "Plate"),
            Relationship("Bread", "on", "Plate"),
            Relationship("Fork", "on", "Plate"),
            Relationship("Plate", "on", "DiningTable"),
        ],
        exceptions=[
            MotionErrorType.CollisionError,
            MotionErrorType.BlockError,
            MotionErrorType.StabilityError,
        ],
    ),
    "working_in_bedroom": HouseholdTask(
        task_name="working_in_bedroom",
        room_includes=["Bedroom", "Livingroom"],
        filter_object_init_room={
            "Bedroom": ["AlarmClock", "CD", "Desk", "Pencil"],
            "Livingroom": ["Laptop"],
        },
        relations=[
            Relationship("AlarmClock", "on", "Desk"),
            Relationship("CD", "on", "Desk"),
            Relationship("Laptop", "on", "Desk"),
            Relationship("Pencil", "on", "Desk"),
        ],
        exceptions=[
            MotionErrorType.CollisionError,
            MotionErrorType.BlockError,
            MotionErrorType.StabilityError,
        ],
    ),
    "movie_and_snack_night": HouseholdTask(
        task_name="movie_and_snack_night",
        room_includes=["Livingroom", "Kitchen"],
        filter_object_init_room={
            "Livingroom": ["Sofa", "DiningTable", "RemoteControl"],
            "Kitchen": ["Plate", "Bread"],
        },
        relations=[
            Relationship("RemoteControl", "on", "Sofa"),
            Relationship("Bread", "on", "Plate"),
            Relationship("Plate", "on", "DiningTable"),
        ],
        exceptions=[
            MotionErrorType.CollisionError,
            MotionErrorType.BlockError,
            MotionErrorType.StabilityError,
        ],
    ),
    "make_tea_and_relax": HouseholdTask(
        task_name="make_tea_and_relax",
        room_includes=["Kitchen", "Livingroom"],
        filter_object_init_room={
            "Kitchen": ["Kettle", "Cup"],
            "Livingroom": ["DiningTable", "RemoteControl", "Sofa"],
        },
        relations=[
            Relationship("Kettle", "on", "CounterTop"),
            Relationship("Cup", "on", "DiningTable"),
            Relationship("RemoteControl", "on", "Sofa"),
        ],
        exceptions=[
            MotionErrorType.CollisionError,
            MotionErrorType.BlockError,
            MotionErrorType.StabilityError,
        ],
    ),
    "prepare_bath": HouseholdTask(
        task_name="prepare_bath",
        room_includes=["Bathroom", "Bedroom"],
        filter_object_init_room={
            "Bathroom": ["Faucet", "SoapBottle"],
            "Bedroom": ["Cloth"],
        },
        relations=[
            Relationship("SoapBottle", "on", "Faucet"),
            Relationship("Cloth", "on", "Faucet"),
        ],
        exceptions=[
            MotionErrorType.CollisionError,
            MotionErrorType.BlockError,
            MotionErrorType.StabilityError,
        ],
    ),
}
