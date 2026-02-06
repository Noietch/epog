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
class ObjectRoomLocation(BaseModel):
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
                    "description": "The room with the highest probability of containing the object.",
                },
            },
            "additionalProperties": False,
            "required": ["steps", "final_answer"],
        }


def get_most_likely_room(
    object_category: str, room_list: list[str]
) -> tuple[str, float]:
    system_message = "You are a household robot. You are asked to analyze one of problems for task planning."
    user_message = f"You need to search for a {object_category} in house, Analyze the probability of {object_category} in {room_list}, give me the most likely room."
    json_schema = {
        "name": "probability_analysis",
        "schema": ObjectRoomLocation.tojson(),
    }
    chat_gpt = ChatGPT()

    max_tries = 3
    for _ in range(max_tries):
        try:
            response = chat_gpt.get_response_text(
                system_message, user_message, json_schema
            )
            location = ObjectRoomLocation.parse_raw(response)
            break
        except ValidationError as e:
            logger.warning(f"{e} retrying...")
            continue
    return location.final_answer


if __name__ == "__main__":
    logger.info(
        get_most_likely_room(
            "shampoo", ["kitchen", "livingroom", "bedroom", "bathroom"]
        )
    )
