from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class Action:
    name: str
    description: str
    function_callable: Callable[..., Any]


@dataclass
class ActionExample:
    user_prompt: str
    expected_action_text: str
