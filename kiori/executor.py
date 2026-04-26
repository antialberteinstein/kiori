from typing import Any, Dict, List, Optional
from .models import Action


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
