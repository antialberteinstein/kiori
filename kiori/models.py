from dataclasses import dataclass
from typing import Callable, Any, Optional


@dataclass
class Action:
    name: str
    description: str
    function_callable: Callable[..., Any]


@dataclass
class ActionExample:
    user_prompt: str
    expected_action_text: Optional[str] = None
    action_name: Optional[str] = None
    kwargs: Optional[dict] = None

    def __post_init__(self):
        # Nếu người dùng truyền action_name và kwargs, tự tạo chuỗi format chuẩn
        if self.action_name and self.expected_action_text is None:
            import json
            from .parser import ACTION_FORMAT
            args_str = json.dumps(self.kwargs) if self.kwargs else "{}"
            self.expected_action_text = ACTION_FORMAT.format(self.action_name, args_str)

