"""Base class for LLM-based planning algorithms."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from loguru import logger

from pog.planning.action import Action, ActionType


class BaseLLMPlanner(ABC):
    """Base class providing common LLM parsing and graph output utilities."""

    @staticmethod
    def get_json_from_llm(llm_output: str) -> str:
        """Extract JSON from LLM output.

        Args:
            llm_output: Raw LLM response text.

        Returns:
            Extracted JSON string.
        """
        json_start = llm_output.find("{")
        json_end = llm_output.rfind("}")
        return llm_output[json_start : json_end + 1]

    @staticmethod
    def is_valid_json(json_str: str) -> bool:
        """Check if string is valid JSON.

        Args:
            json_str: String to validate.

        Returns:
            True if valid JSON, False otherwise.
        """
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            return False

    @staticmethod
    def create_action_from_raw_simple(raw_action: str) -> Action | None:
        """Parse action from simple format: Action(id1, id2) or Action(id).

        Args:
            raw_action: Action string like "Pick(2, 5)" or "Open(3)".

        Returns:
            Action object or None if parsing fails.
        """
        # Try two-argument pattern first: Pick(2, 5)
        pattern = r"(\w+)\((\d+),\s*(\d+)\)"
        match = re.match(pattern, raw_action)
        if match:
            action_type_str = match.group(1)
            object_id1 = int(match.group(2))
            object_id2 = int(match.group(3))

            if action_type_str == "Pick":
                return Action(
                    action_type=ActionType.Pick,
                    edge_edit_pair=((object_id2, object_id1), None),
                )
            elif action_type_str == "Place":
                return Action(
                    action_type=ActionType.Place,
                    edge_edit_pair=(None, (object_id2, object_id1)),
                )

        # Try single-argument pattern: Open(3)
        pattern = r"(\w+)\((\d+)\)"
        match = re.match(pattern, raw_action)
        if match:
            action_type_str = match.group(1)
            object_id = int(match.group(2))

            action_map = {
                "Open": ActionType.Open,
                "Close": ActionType.Close,
                "Walk": ActionType.Walk,
            }
            if action_type_str in action_map:
                return Action(
                    action_type=action_map[action_type_str],
                    edge_edit_pair=((object_id, object_id), None),
                )

        logger.warning(f"Failed to parse action: {raw_action}")
        return None

    @staticmethod
    def create_action_from_raw_verbose(raw_action: str) -> Action | None:
        """Parse action from verbose format: Action(id name, id name).

        Args:
            raw_action: Action string like "Pick(2 Box, 5 Garage)".

        Returns:
            Action object or None if parsing fails.
        """
        pattern = r"(\w+)\((\d+)\s\w+,\s(\d+)\s\w+\)"
        match = re.match(pattern, raw_action)
        if not match:
            return None

        action_type_str = match.group(1)
        object_id1 = int(match.group(2))
        object_id2 = int(match.group(3))

        if action_type_str == "Pick":
            return Action(
                action_type=ActionType.Pick,
                edge_edit_pair=((object_id2, object_id1), None),
            )
        elif action_type_str == "Place":
            return Action(
                action_type=ActionType.Place,
                edge_edit_pair=(None, (object_id2, object_id1)),
            )
        return None

    def json_to_actions(
        self, json_data: dict[str, list], verbose: bool = False
    ) -> list[Action]:
        """Convert JSON plan data to list of Action objects.

        Args:
            json_data: Dictionary with "Plan" key containing action strings.
            verbose: Use verbose format parsing if True.

        Returns:
            List of Action objects.
        """
        raw_plan = json_data["Plan"]
        parser = (
            self.create_action_from_raw_verbose
            if verbose
            else self.create_action_from_raw_simple
        )
        actions = []
        for raw_action in raw_plan:
            action = parser(raw_action)
            if action is not None:
                actions.append(action)
        return actions

    def output_belief_graph(self) -> str:
        """Output belief graph as human-readable string.

        Returns:
            String representation of belief graph edges.
        """
        output = ""
        for edge in self.belif_graph.edge_dict.values():
            name_item1 = self.belif_graph.node_dict[edge.child_id]
            name_item2 = self.belif_graph.node_dict[edge.parent_id]
            s = f"{name_item1} is {edge.semantic_info['relationType']} {name_item2}"
            output += s + "\n"
        return output

    def output_task_graph(self) -> str:
        """Output task graph as human-readable string.

        Returns:
            String representation of task graph edges.
        """
        output = ""
        for edge in self.task_graph.edge_dict.values():
            name_item1 = self.task_graph.node_dict[edge.child_id]
            name_item2 = self.task_graph.node_dict[edge.parent_id]
            s = f"{name_item1} is {edge.semantic_info['relationType']} {name_item2}"
            output += s + "\n"
        return output

    def parse_llm_response(
        self, response: str, verbose: bool = False
    ) -> list[Action] | None:
        """Parse LLM response into actions.

        Args:
            response: Raw LLM response text.
            verbose: Use verbose format parsing if True.

        Returns:
            List of actions or None if parsing fails.
        """
        json_str = self.get_json_from_llm(response)

        if not self.is_valid_json(json_str):
            logger.warning("LLM output is not valid JSON")
            return None

        try:
            json_data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("LLM output is not valid JSON")
            return None

        try:
            actions = self.json_to_actions(json_data, verbose=verbose)
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"LLM output is not valid action: {e}")
            return None

        logger.info("LLM-generated plan:")
        for action in actions:
            logger.info(f"  {action}")

        return actions

    @abstractmethod
    def main_loop(self) -> None:
        """Main planning loop - to be implemented by subclasses."""
        pass
