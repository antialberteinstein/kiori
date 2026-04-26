from typing import List, Optional
from .models import Action, ActionExample
from .memory import MilvusLTM, ReplayBuffer
from .router import MarkovRouter


class KioriAgent:
    def __init__(
        self,
        ltm: Optional[MilvusLTM] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        router: Optional[MarkovRouter] = None,
        alpha: float = 0.5,
        beta: float = 0.5
    ) -> None:
        self.actions: List[Action] = []
        self.examples: List[ActionExample] = []
        self.ltm = ltm
        self.replay_buffer = replay_buffer
        self.router = router
        self.alpha = alpha
        self.beta = beta
        self.previous_action: Optional[str] = None

    def add_action(self, action: Action) -> None:
        self.actions.append(action)
        if self.router and action.name not in self.router.all_actions:
            self.router.all_actions.append(action.name)

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

            if self.router:
                combined_results = []
                for ex, cosine_score in search_results:
                    action_name = ex.expected_action_text.split("(")[0].strip()
                    combined_score = self.router.calculate_combined_score(
                        action_name=action_name,
                        cosine_score=cosine_score,
                        previous_action=self.previous_action,
                        alpha=self.alpha,
                        beta=self.beta
                    )
                    combined_results.append((ex, combined_score))
                search_results = combined_results

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

        if ctx:
            # Simulate execution for next turn routing
            action_text = ctx[0].expected_action_text
            self.previous_action = action_text.split("(")[0].strip()
