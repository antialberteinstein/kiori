from typing import List, Dict, Optional
from .models import Action, ActionExample
from .parser import ACTION_FORMAT_INSTRUCTION_EN, ACTION_FORMAT_INSTRUCTION_VI, ACTION_FORMAT_HINT


# ============================================================
# Phần 1: Format NỘI DUNG prompt (không phụ thuộc chat template)
# ============================================================

def get_system_prompt(actions: Optional[List[Action]] = None, lang: str = "en") -> str:
    """Tạo system prompt chuẩn chứa hướng dẫn chi tiết và danh sách hành động."""
    if lang == "vi":
        system_prompt = (
            "Hệ thống: Bạn là một trợ lý AI thông minh chuyên điều khiển hành động.\n"
            "QUY TẮC:\n"
            "1. Chỉ sử dụng các hành động được liệt kê bên dưới.\n"
            f"2. Luôn xuất ra kết quả theo định dạng: {ACTION_FORMAT_INSTRUCTION_VI}\n"
            "3. Không giải thích thêm, không chào hỏi.\n\n"
        )
        if actions:
            system_prompt += "DANH SÁCH HÀNH ĐỘNG KHẢ DỤNG:\n"
            for act in actions:
                system_prompt += f"- {act.name}: {act.description}\n"
            system_prompt += "\n"
    else:
        system_prompt = (
            "System: You are an intelligent agent that executes actions based on user commands.\n"
            "RULES:\n"
            "1. ONLY use the actions listed below.\n"
            f"2. ALWAYS output in the format: {ACTION_FORMAT_INSTRUCTION_EN}\n"
            "3. DO NOT provide any conversation, greetings, or explanations.\n\n"
        )
        if actions:
            system_prompt += "AVAILABLE ACTIONS:\n"
            for act in actions:
                system_prompt += f"- {act.name}: {act.description}\n"
            system_prompt += "\n"
            
    return system_prompt.strip()

def get_action_not_found_observation(action_name: str, valid_names: str, lang: str = "en") -> str:
    """Tạo tin nhắn hệ thống khi mô hình dự đoán một hành động không tồn tại."""
    if lang == "vi":
        return (
            f"[System Observation: Hành động '{action_name}' không tồn tại. "
            f"Các hành động hợp lệ là: {valid_names}. Hãy chọn hành động đúng.]"
        )
    return (
        f"[System Observation: Action '{action_name}' does not exist. "
        f"Valid actions are: {valid_names}. Please choose a correct action.]"
    )

def get_broken_format_observation(lang: str = "en") -> str:
    """Tạo tin nhắn hệ thống khi mô hình trả về sai định dạng."""
    if lang == "vi":
        return (
            "[System Observation: Văn bản của bạn bị sai định dạng. "
            f"Hãy sinh lại chỉ dùng định dạng {ACTION_FORMAT_HINT}]"
        )
    return (
        "[System Observation: Your text format is incorrect. "
        f"Please generate again using only the format {ACTION_FORMAT_HINT}]"
    )


def get_summarize_observation_prompt(user_prompt: str, action_result: str, lang: str = "en") -> str:
    """Tạo prompt để LLM tổng hợp kết quả action thành câu trả lời tự nhiên cho user."""
    if lang == "vi":
        return (
            f"Câu hỏi của người dùng: {user_prompt}\n"
            f"Kết quả quan sát: {action_result}\n\n"
            f"Hãy trả lời người dùng một cách tự nhiên và ngắn gọn. "
            f"Không đề cập đến tên hành động hay chi tiết kỹ thuật:"
        )
    return (
        f"User question: {user_prompt}\n"
        f"Observed result: {action_result}\n\n"
        f"Answer the user naturally and concisely. "
        f"Do not mention action names or technical details:"
    )


# ============================================================
# Phần 2: Chat Templates (dùng khi chat_format được chỉ định)
# ============================================================

def gemma_template(messages: List[Dict[str, str]], model_prefix: str = "") -> str:
    prompt = "<bos>"
    system_text = ""
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        # Gemma không có role system riêng rẽ, và không chấp nhận 2 lượt user liên tiếp.
        # Do đó, ta gộp system prompt vào ngay lượt user đầu tiên.
        if role == "system":
            system_text = content + "\n\n"
        elif role == "user":
            prompt += f"<start_of_turn>user\n{system_text}{content}<end_of_turn>\n"
            system_text = "" # Chỉ gộp 1 lần
        elif role == "assistant":
            prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"
    
    # Kích hoạt model trả lời (với prefix nếu có)
    prompt += f"<start_of_turn>model\n{model_prefix}"
    return prompt

def llama3_template(messages: List[Dict[str, str]], model_prefix: str = "") -> str:
    prompt = "<|begin_of_text|>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    
    # Kích hoạt model trả lời (với prefix nếu có)
    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{model_prefix}"
    return prompt

def chatml_template(messages: List[Dict[str, str]], model_prefix: str = "") -> str:
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    # Kích hoạt model trả lời (với prefix nếu có)
    prompt += f"<|im_start|>assistant\n{model_prefix}"
    return prompt

TEMPLATES = {
    "gemma": gemma_template,
    "llama3": llama3_template,
    "chatml": chatml_template,
}

def apply_chat_template(messages: List[Dict[str, str]], template_name: str = "gemma", model_prefix: str = "") -> str:
    """
    Áp dụng Chat Template cho một list messages.
    Hỗ trợ: gemma (mặc định), llama3, chatml.
    model_prefix: Chuỗi được điền sẵn vào đầu lượt model (prefix-fill) để hướng SLM sinh đúng format.
    """
    if template_name not in TEMPLATES:
        raise ValueError(f"Chat template '{template_name}' is not supported. Available templates: {list(TEMPLATES.keys())}")
    return TEMPLATES[template_name](messages, model_prefix=model_prefix)
