import json
import random
from typing import List, Optional, Callable, Any
from .models import Action, ActionExample
from .memory import MilvusLTM, ReplayBuffer
from .executor import execute_action
from .parser import KioriParser, ParseStatus, ACTION_PREFIX, ACTION_FORMAT
from .chat_templates import (
    apply_chat_template,
    get_system_prompt,
    get_action_not_found_observation,
    get_broken_format_observation,
    get_summarize_observation_prompt
)

def context_shuffler(examples: List[ActionExample]) -> List[ActionExample]:
    shuffled = list(examples)
    random.shuffle(shuffled)
    return shuffled


def format_prompt(user_query: str, examples: List[ActionExample], actions: Optional[List[Action]] = None, lang: str = "en") -> str:
    """
    Format prompt thuần túy theo định dạng User:... Action:...
    Trả về chuỗi nội dung, không chứa chat template hay parser tokens.
    """
    prompt = get_system_prompt(actions, lang=lang)
    
    # Few-shot examples theo format User-Action
    if examples:
        prompt += "\n\nExamples:\n"
        for ex in examples:
            prompt += f"User: {ex.user_prompt}\nAction: {ex.expected_action_text}\n\n"
    
    # User query hiện tại
    prompt += f"User: {user_query}"
    
    return prompt



def _apply_template(prompt: str, chat_format: str) -> str:
    """Áp dụng chat template + prefix-fill [ACTION: vào prompt."""
    messages = [{"role": "user", "content": prompt}]
    return apply_chat_template(messages, chat_format, model_prefix=ACTION_PREFIX)


class KioriAgent:
    DEBUG = False

    def __init__(
        self,
        ltm: Optional[MilvusLTM] = None,
        replay_buffer: Optional[ReplayBuffer] = None,
        chat_format: Optional[str] = None,
        lang: str = "en",
        threshold: float = 0.5,
        max_copies: int = 3,
        sample_n: int = 3,
    ) -> None:
        """
        Args:
            chat_format: Tên chat template ("gemma", "llama3", "chatml").
                         Nếu None, Kiori trả prompt thuần cho callback tự xử lý.
                         Nếu được set, Kiori tự apply template + prefix-fill.
        """
        self.actions: List[Action] = []
        self.ltm = ltm
        self.replay_buffer = replay_buffer
        self.chat_format = chat_format
        self.lang = lang

        self.threshold = threshold
        self.max_copies = max_copies
        self.sample_n = sample_n

    def add_action(self, action: Action) -> None:
        self.actions.append(action)

    def get_context_examples(
        self,
        user_prompt: str,
    ) -> List[ActionExample]:
        merged: List[ActionExample] = []

        # 1. Lấy từ Long-Term Memory (Similarity Search)
        if self.ltm:
            search_results = self.ltm.search(user_prompt, top_k=5)
            scaled = self.ltm.scale_examples(
                search_results, threshold=self.threshold, max_copies=self.max_copies
            )
            merged.extend(scaled)

        # 2. Lấy từ Replay Buffer (Experience Replay)
        if self.replay_buffer:
            sampled = self.replay_buffer.sample_buffer(n=self.sample_n)
            merged.extend(sampled)
        
        if KioriAgent.DEBUG:
            print('[DEBUG] Merged samples:\n')
            for i, sample in enumerate(merged):
                print(f"[DEBUG] #{i}: {sample.user_prompt}")
            print('[DEBUG] \n----------------')

        return merged

    def _prepare_prompt(self, user_prompt: str, ctx: List[ActionExample]) -> str:
        """Tạo prompt hoàn chỉnh, có hoặc không có chat template."""
        shuffled_ctx = context_shuffler(ctx)
        raw_prompt = format_prompt(user_prompt, shuffled_ctx, self.actions, lang=self.lang)
        
        if self.chat_format:
            # Kiori tự apply chat template + prefix-fill [ACTION:
            return _apply_template(raw_prompt, self.chat_format)
        else:
            # Thêm 'Action:' để hướng model sinh action thay vì tiếp tục pattern
            return raw_prompt + "\nAction:"

    def _parse_response(self, llm_response: str, parser: KioriParser):
        """Parse response, tự động ghép ACTION_PREFIX nếu agent đã dùng chat_format."""
        if self.chat_format:
            # Kiori đã prefix-fill [ACTION:, cần ghép lại
            full = ACTION_PREFIX + llm_response if not llm_response.startswith(ACTION_PREFIX) else llm_response
            return parser.parse(full)
        else:
            return parser.parse(llm_response)

    def run(
        self,
        user_prompt: str,
        llm_callback: Callable[[str], str],
        max_retries: int = 3,
        summarize_observation: bool = False
    ) -> Any:
        ctx = self.get_context_examples(user_prompt)
        current_prompt = self._prepare_prompt(user_prompt, ctx)

        parser = KioriParser()
        retries = 0

        while retries < max_retries:
            if KioriAgent.DEBUG:
                print(f"[DEBUG] === PROMPT SENT TO LLM ===\n{current_prompt}\n[DEBUG] === END PROMPT ===")
            
            llm_response = llm_callback(current_prompt)
            parse_result = self._parse_response(llm_response, parser)

            if parse_result.status == ParseStatus.SUCCESS:
                # Check if action exists
                action_exists = any(a.name == parse_result.action_name for a in self.actions)
                if not action_exists:
                    retries += 1
                    valid_names = ", ".join(a.name for a in self.actions)
                    observation = get_action_not_found_observation(parse_result.action_name, valid_names, lang=self.lang)
                    
                    if self.chat_format:
                        # Rebuild prompt with observation context
                        current_prompt = self._prepare_prompt(user_prompt, ctx)
                    else:
                        current_prompt += f"\nAction: {llm_response}\n{observation}\nUser: {user_prompt}\nAction:"
                    continue

                try:
                    result = execute_action(parse_result.action_name, parse_result.kwargs, self.actions)
                except TypeError as e:
                    # Model sinh action đúng nhưng thiếu/sai ARGS → retry
                    retries += 1
                    observation = get_broken_format_observation(lang=self.lang)
                    if self.chat_format:
                        current_prompt = self._prepare_prompt(user_prompt, ctx)
                    else:
                        current_prompt += f"\nAction: {llm_response}\n{observation}\nUser: {user_prompt}\nAction:"
                    continue

                if KioriAgent.DEBUG:
                    print(f"[DEBUG] Action: {parse_result.action_name}")
                    print(f"[DEBUG] Observation: {result}")
                    print("[DEBUG] ----------------")

                if self.replay_buffer:
                    args_str = json.dumps(parse_result.kwargs) if parse_result.kwargs else "{}"
                    action_text = ACTION_FORMAT.format(parse_result.action_name, args_str)
                    new_example = ActionExample(
                        user_prompt=user_prompt,
                        expected_action_text=action_text
                    )
                    self.replay_buffer.update_buffer([new_example])

                # Tổng hợp kết quả thành câu trả lời tự nhiên nếu được bật
                if summarize_observation:
                    summary_prompt = get_summarize_observation_prompt(
                        user_prompt=user_prompt,
                        action_result=str(result),
                        lang=self.lang
                    )
                    if self.chat_format:
                        summary_prompt = apply_chat_template(
                            [{"role": "user", "content": summary_prompt}],
                            self.chat_format
                        )
                    return llm_callback(summary_prompt)

                return result

            elif parse_result.status == ParseStatus.NATURAL_CHAT:
                # Không lưu kết quả rỗng vào ReplayBuffer
                if self.replay_buffer and parse_result.raw_text.strip():
                    new_example = ActionExample(
                        user_prompt=user_prompt,
                        expected_action_text=parse_result.raw_text
                    )
                    self.replay_buffer.update_buffer([new_example])
                return parse_result.raw_text

            elif parse_result.status == ParseStatus.BROKEN_FORMAT:
                retries += 1
                observation = get_broken_format_observation(lang=self.lang)
                if self.chat_format:
                    current_prompt = self._prepare_prompt(user_prompt, ctx)
                else:
                    current_prompt += f"\nAction: {llm_response}\n{observation}\nUser: {user_prompt}\nAction:"

        raise ValueError(
            f"Agent failed to produce a valid action format after {max_retries} retries."
        )
