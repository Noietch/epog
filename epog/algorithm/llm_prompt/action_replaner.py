from loguru import logger
from pydantic import BaseModel, ValidationError

from epog.algorithm.epog.fake_simulator import MotionError, MotionErrorType
from epog.utils.gpt_helper import ChatGPT
from pog.planning.action import Action, ActionType


class Step(BaseModel):
    explanation: str
    output: str

    def tojson():
        return {
            "type": "object",
            "properties": {
                "explanation": {"type": "string"},
                "output": {"type": "string"},
            },
            "description": "Step-by-step analysis of the action.",
            "additionalProperties": False,
        }


class ActionStep(BaseModel):
    action: str

    def tojson():
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                }
            },
            "description": "Action to resolve the error.",
            "additionalProperties": False,
        }


# object location
class ActionSeq(BaseModel):
    steps: list[Step]
    final_answer: list[ActionStep]

    def tojson():
        return {
            "strict": False,
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": Step.tojson(),
                },
                "final_answer": {
                    "type": "array",
                    "items": ActionStep.tojson(),
                    "description": "action sequence to resolve the error.",
                },
            },
            "additionalProperties": False,
            "required": ["steps", "final_answer"],
        }


def get_resolve_action_seq(error: MotionError) -> ActionSeq:
    system_message = "You are a robot, and you need to insert some action to resolved any errors that occur during the task planning process."
    user_message = f"""
        You are going to resolve the error that occurs during the task planning process.
        The primitives are: Pick(x, y): Pick x from y, Place(x, y): Place x on y, Open(x): Open x, Close(x): Close x
        The parking place is {error.parking_place}, you can place temporarily place the object here.
        Here is the example:
        error: "place 1 on 2 is failure and object 1 will be collision with [3]"
        Whole Anaylsis: object 1 is collision within object 3, so i need to adjust the object 3 so that we can place object 1 on object 2.
            Robot Hand State: because the failed action type is place, so my hand is occupied, object 1 is in my hand.
            Frist Step: Because object 1 is in my hand, so i need to place object 1 on the parking place. Place(1, 0)
            Second Step: Because object 3 is collision with object 1, so i need to remove the collision, Pick(3, 2)
            Third Step: I need to replace object 3 in the parking place so that i can pick the object 1, Place(3, 0)
            Fourth Step: I need to pick object 1 from the parking place, Pick(1, 0)
            Fifth Step: I need to replay the failed action, place object 1 on object 2, Place(1, 2)
            then I summarize the action sequence below:
        action sequence: ["Place(1, 0)", "Pick(3, 2)", "Place(3, 0)", "Pick(1, 0)", "Place(1, 2)"]
        Your robot is trying to {error.failure_action} but it failed. The error is {error}.
    """
    json_schema = {"name": "probability_analysis", "schema": ActionSeq.tojson()}
    chat_gpt = ChatGPT()
    response = chat_gpt.get_response_text(system_message, user_message, json_schema)
    while True:
        try:
            action_seq = ActionSeq.parse_raw(response)
            break
        except ValidationError as e:
            logger.warning(f"{e} {response}")
            response = chat_gpt.get_response_text(
                system_message, user_message, json_schema
            )
    actions = []
    for step in action_seq.final_answer:
        actions.append(Action.from_func_string(step.action))
    return actions


if __name__ == "__main__":
    error = MotionError(
        "Pick 1 from 2 is failure and object 3 on 1 so it is unstable",
        MotionErrorType.StabilityError,
        Action(((2, 1), None), action_type=ActionType.Pick),
        [3],
        "3 on 1",
    )
    logger.info(get_resolve_action_seq(error))
