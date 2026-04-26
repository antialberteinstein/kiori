import json
import random
from typing import List, Optional, Callable, Any
from .models import Action, ActionExample
from .memory import MilvusLTM, ReplayBuffer
from .executor import parse_llm_output, execute_action


def context_shuffler(examples: List[ActionExample]) -> List[ActionExample]:
    shuffled = list(examples)
    random.shuffle(shuffled)
    return shuffled


def format_prompt(user_query: str, examples: List[ActionExample]) -> str:
    prompt = "System: You are an intelligent agent. " \
             "Based on the examples, output the correct action " \
             "in format [ACTION: name, ARGS: {...}]\n"
    if examples:
        prompt += "Examples:\n"
        for ex in examples:
            prompt += (
                f"User: {ex.user_prompt}\n"
                f"Action: {ex.expected_action_text}\n\n"
            )
    prompt += f"User: {user_query}\nAction:"
    return prompt


class KioriAgent:
    def __init__(
        self,
        ltm: Optional[MilvusLTM] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
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

    def run(self, user_prompt: str, llm_callback: Callable[[str], str]) -> Any:
        ctx = self.get_context_examples(user_prompt)
        shuffled_ctx = context_shuffler(ctx)
        prompt = format_prompt(user_prompt, shuffled_ctx)

        llm_response = llm_callback(prompt)

        action_name, kwargs = parse_llm_output(llm_response)
        result = execute_action(action_name, kwargs, self.actions)

        if action_name:
            if self.replay_buffer:
                args_str = json.dumps(kwargs) if kwargs else "{}"
                action_text = f"[ACTION: {action_name}, ARGS: {args_str}]"
                new_example = ActionExample(
                    user_prompt=user_prompt,
                    expected_action_text=action_text
                )
                self.replay_buffer.update_buffer([new_example])

        return result
