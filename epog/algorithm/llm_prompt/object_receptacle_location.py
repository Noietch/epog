from loguru import logger
from pydantic import BaseModel, ValidationError

from epog.utils.gpt_helper import ChatGPT


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
            "description": "Step-by-step analysis of the probability of an object in each room.",
            "additionalProperties": False,
        }


# object location
class ObjectReceptacleLocation(BaseModel):
    steps: list[Step]
    final_answer: str

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
                    "type": "string",
                    "description": "The receptacle with the highest probability of containing the object.",
                },
            },
            "additionalProperties": False,
            "required": ["steps", "final_answer"],
        }


def get_most_likely_receptacle(
    object_category: str, room: str, receptcle: list[str]
) -> ObjectReceptacleLocation:
    system_message = "You are a household robot. You are asked to analyze one of problems for task planning."
    user_message = f"You need to search for a {object_category} in {room}. The room has the following receptcle objects: {receptcle}. \
            Based on typical household arrangements and the nature of the item, which receptcle object is most likely to support/contain the item."
    json_schema = {
        "name": "probability_analysis",
        "schema": ObjectReceptacleLocation.tojson(),
    }
    chat_gpt = ChatGPT()
    response = chat_gpt.get_response_text(system_message, user_message, json_schema)
    while True:
        try:
            location = ObjectReceptacleLocation.parse_raw(response)
            break
        except ValidationError as e:
            logger.warning(f"{e}")
            response = chat_gpt.get_response_text(
                system_message, user_message, json_schema
            )
    return location.final_answer


if __name__ == "__main__":
    logger.info(
        get_most_likely_receptacle("apple", "kitchen", ["fridge", "dinningtable"])
    )
