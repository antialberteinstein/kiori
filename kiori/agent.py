from typing import List, Optional
from .models import Action, ActionExample
from .memory import MilvusLTM, ReplayBuffer


class KioriAgent:
    def __init__(
        self,
        ltm: Optional[MilvusLTM] = None,
        replay_buffer: Optional[ReplayBuffer] = None
    ) -> None:
        self.actions: List[Action] = []
        self.examples: List[ActionExample] = []
        self.ltm = ltm
        self.replay_buffer = replay_buffer

    def add_action(self, action: Action) -> None:
        self.actions.append(action)

    def add_example(self, example: ActionExample) -> None:
        self.examples.append(example)

    def get_context_examples(
        self,
        user_prompt: str,
        threshold: float = 0.5,
        max_copies: int = 3,
        sample_n: int = 3
    ) -> List[ActionExample]:
        merged: List[ActionExample] = []

        if self.ltm:
            search_results = self.ltm.search(user_prompt, top_k=5)
            scaled = self.ltm.scale_examples(
                search_results, threshold=threshold, max_copies=max_copies
            )
            merged.extend(scaled)

        if self.replay_buffer:
            sampled = self.replay_buffer.sample_buffer(n=sample_n)
            merged.extend(sampled)

        return merged

    def run(self, user_prompt: str) -> None:
        ctx = self.get_context_examples(user_prompt)
        print(f"Prompt received: {user_prompt}")
        print(f"Context examples count: {len(ctx)}")
