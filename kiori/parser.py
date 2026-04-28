import re
import json
from enum import Enum, auto
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


# ============================================================
# Action Format Constants
# Tập trung mọi biến thể của cấu trúc [ACTION:...] tại đây.
# Các file khác import constants này và dùng .format() khi cần.
# ============================================================

# Prefix dùng cho prefix-fill khi áp chat template nội bộ
ACTION_PREFIX = "[ACTION:"

# Template đầy đủ cho runtime: format(action_name, args_str)
ACTION_FORMAT = "[ACTION: {}, ARGS: {}]"

# Mô tả format cho system prompt EN
ACTION_FORMAT_INSTRUCTION_EN = "[ACTION: name, ARGS: {key: value}]"

# Mô tả format cho system prompt VI
ACTION_FORMAT_INSTRUCTION_VI = "[ACTION: tên_hành_động, ARGS: {tham_số}]"

# Mô tả format cho observation/error messages
ACTION_FORMAT_HINT = "[ACTION: name, ARGS: {...}]"

# Regex pattern để parse action format
ACTION_REGEX = r"\[ACTION:\s*([a-zA-Z0-9_-]+)(?:,\s*ARGS:\s*(\{.*?\}))?\]"

# Broken format detection keywords
ACTION_KEYWORD = "[ACTION:"
ARGS_KEYWORD = "ARGS:"


class ParseStatus(Enum):
    """Represents the status of parsing the LLM output."""
    SUCCESS = auto()
    BROKEN_FORMAT = auto()
    NATURAL_CHAT = auto()


@dataclass
class ParseResult:
    """Contains the parsed outcome from the KioriParser."""
    status: ParseStatus
    action_name: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""


class KioriParser:
    """
    Intelligently parses LLM text outputs into actionable commands or conversational text.
    """

    def parse(self, text: str) -> ParseResult:
        """
        Parses the raw string from an LLM.

        Args:
            text (str): The raw text response.

        Returns:
            ParseResult: A structured object defining whether parsing was successful,
                         broken, or simply natural chat.
        """
        # 1. Check for perfect format match
        match = re.search(ACTION_REGEX, text, re.IGNORECASE | re.DOTALL)
        if match:
            action_name = match.group(1).strip()
            args_str = match.group(2)
            kwargs = {}
            if args_str:
                try:
                    kwargs = json.loads(args_str)
                except json.JSONDecodeError:
                    return ParseResult(status=ParseStatus.BROKEN_FORMAT, raw_text=text)
            return ParseResult(
                status=ParseStatus.SUCCESS,
                action_name=action_name,
                kwargs=kwargs,
                raw_text=text
            )

        # 2. Check for broken format indicators
        upper_text = text.upper()
        if ACTION_KEYWORD in upper_text or ARGS_KEYWORD in upper_text:
            return ParseResult(status=ParseStatus.BROKEN_FORMAT, raw_text=text)

        # 3. Otherwise, it's natural chat
        return ParseResult(status=ParseStatus.NATURAL_CHAT, raw_text=text)
