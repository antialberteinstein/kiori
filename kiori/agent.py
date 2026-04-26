import json
import random
from typing import List, Optional, Callable, Any
from .models import Action, ActionExample
from .memory import MilvusLTM, ReplayBuffer
from .executor import execute_action
from .parser import KioriParser, ParseStatus


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

    def run(
        self,
        user_prompt: str,
        llm_callback: Callable[[str], str],
        max_retries: int = 3
    ) -> Any:
        ctx = self.get_context_examples(user_prompt)
        shuffled_ctx = context_shuffler(ctx)
        base_prompt = format_prompt(user_prompt, shuffled_ctx)

        parser = KioriParser()
        current_prompt = base_prompt
        retries = 0

        while retries < max_retries:
            llm_response = llm_callback(current_prompt)
            parse_result = parser.parse(llm_response)

            if parse_result.status == ParseStatus.SUCCESS:
                result = execute_action(parse_result.action_name, parse_result.kwargs, self.actions)

                if self.replay_buffer:
                    args_str = json.dumps(parse_result.kwargs) if parse_result.kwargs else "{}"
                    action_text = f"[ACTION: {parse_result.action_name}, ARGS: {args_str}]"
                    new_example = ActionExample(
                        user_prompt=user_prompt,
                        expected_action_text=action_text
                    )
                    self.replay_buffer.update_buffer([new_example])

                return result

            elif parse_result.status == ParseStatus.NATURAL_CHAT:
                if self.replay_buffer:
                    new_example = ActionExample(
                        user_prompt=user_prompt,
                        expected_action_text=parse_result.raw_text
                    )
                    self.replay_buffer.update_buffer([new_example])
                return parse_result.raw_text

            elif parse_result.status == ParseStatus.BROKEN_FORMAT:
                retries += 1
                observation = (
                    "\n[System Observation: Text của bạn bị sai định dạng. "
                    "Hãy sinh lại chỉ dùng định dạng [ACTION: name, ARGS: {...}]]"
                )
                current_prompt = current_prompt + "\n" + llm_response + observation

        raise ValueError(
            f"Agent failed to produce a valid action format after {max_retries} retries."
        )
