import re
import json
from typing import Any, Dict, List, Tuple, Optional
from .models import Action


def parse_llm_output(text: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Parse LLM output containing [ACTION: name, ARGS: {...}]
    """
    match = re.search(
        r"\[ACTION:\s*([^,\]]+)(?:,\s*ARGS:\s*(\{.*?\}))?\]", text
    )
    if match:
        action_name = match.group(1).strip()
        args_str = match.group(2)
        kwargs = {}
        if args_str:
            try:
                kwargs = json.loads(args_str)
            except json.JSONDecodeError:
                pass
        return action_name, kwargs
    return None, {}


def execute_action(
    action_name: Optional[str],
    kwargs: Dict[str, Any],
    actions: List[Action]
) -> Any:
    """
    Execute the action with parsed kwargs
    """
    if not action_name:
        raise ValueError("No action name parsed from LLM output.")

    for action in actions:
        if action.name == action_name:
            return action.function_callable(**kwargs)

    raise ValueError(f"Action '{action_name}' not found.")
