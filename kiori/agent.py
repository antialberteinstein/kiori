from typing import List
from .models import Action, ActionExample


class KioriAgent:
    def __init__(self) -> None:
        self.actions: List[Action] = []
        self.examples: List[ActionExample] = []

    def add_action(self, action: Action) -> None:
        self.actions.append(action)

    def add_example(self, example: ActionExample) -> None:
        self.examples.append(example)

    def run(self, user_prompt: str) -> None:
        print(f"Prompt received: {user_prompt}")
