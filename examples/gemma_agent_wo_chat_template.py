import os
import sys
# Ensure we load the local 'kiori' package, not the one installed via pip
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["USE_TF"] = "0"

import datetime
import torch
import numpy as np

from kiori.agent import KioriAgent
from kiori.models import Action, ActionExample
from kiori.memory import MilvusLTM, ReplayBuffer

print("Loading LLM (unsloth/gemma-3-1b-it)... This might take a while.")
# Initialize the pipeline. We use bfloat16 if available, otherwise float32.
# Gemma models work well with bfloat16.
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# We use the text-generation pipeline.
from transformers import pipeline
generator = pipeline(
    "text-generation",
    model="unsloth/gemma-3-1b-it",
    torch_dtype=dtype,
    device=device
)

def llm_callback(prompt: str) -> str:
    """
    Callback: nhận prompt thuần từ Kiori (không có chat template),
    wrap vào messages và dùng transformers pipeline để áp chat template tự động.
    Thêm 'Action:' vào cuối để hướng model sinh action thay vì tiếp tục pattern.
    """
    messages = [{"role": "user", "content": prompt}]
    
    outputs = generator(messages, max_new_tokens=128, temperature=0.2)
    generated_text = outputs[0]['generated_text'][-1]['content'].strip()
    
    print(f"\n[LLM Raw Output] -> {generated_text}\n")
    
    return generated_text


# --- Define Actions ---

def get_current_time() -> str:
    """Action 1: Get the actual current time."""
    now = datetime.datetime.now()
    return f"The current date and time is: {now.strftime('%Y-%m-%d %H:%M:%S')}"

def calculate_expression(expression: str) -> str:
    """Action 2: Calculate a mathematical expression securely."""
    # We restrict characters to prevent malicious code execution via eval()
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression).issubset(allowed_chars):
        return "Error: Invalid characters in math expression. Only numbers and basic operators are allowed."
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

def list_files() -> str:
    """Action 3: Read and list files in the current working directory."""
    try:
        files = os.listdir(".")
        # Filter out hidden files just to keep the output clean
        visible_files = [f'`{file}`' for file in files if not file.startswith('.')]
        if not visible_files:
            return "The current directory is empty."
        return "Files in current directory: " + ", ".join(visible_files)
    except Exception as e:
        return f"Error reading directory: {str(e)}"



# --- Setup Kiori Agent ---

def main():
    print("\nInitializing Kiori Agent...")
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize Memory modules
    db_path = os.path.join(log_dir, "kiori_gemma_example.db")
    ltm = MilvusLTM(db_path=db_path, collection_name="gemma_examples")
    replay_buffer = ReplayBuffer()

    agent = KioriAgent(ltm=ltm, replay_buffer=replay_buffer, max_copies=3)

    # Register Actions
    agent.add_action(Action("get_current_time", "Fetch the current time", get_current_time))
    agent.add_action(Action("calculate_expression", "Evaluate math expressions", calculate_expression))
    agent.add_action(Action("list_files", "List files in the directory", list_files))

    # Add Few-Shot Examples to Long-Term Memory (Structured data, no parser leak)
    examples = [
        ActionExample("What time is it right now?", action_name="get_current_time", kwargs={}),
        ActionExample("Tell me the current date and time", action_name="get_current_time", kwargs={}),
        
        ActionExample("Calculate 15 * 4", action_name="calculate_expression", kwargs={"expression": "15 * 4"}),
        ActionExample("What is 100 / 25 + 5?", action_name="calculate_expression", kwargs={"expression": "100 / 25 + 5"}),
        ActionExample("I need to know the result of 25 * 18", action_name="calculate_expression", kwargs={"expression": "25 * 18"}),
        
        ActionExample("What files are in this folder?", action_name="list_files", kwargs={}),
        ActionExample("Show me the directory contents", action_name="list_files", kwargs={}),
    ]
    ltm.clear()
    ltm.add_examples(examples)

    print("\n--- Agent Ready! Let's test it ---\n")
    
    # Test Queries
    queries = [
        "Hey, can you check what time it is?",
        "I need to know the result of 25 * 18",
        "Please list all the files located in the current directory."
    ]

    for query in queries:
        print(f"User: {query}")
        try:
            # The agent handles context fetching, prompt formatting, LLM execution, 
            # parsing, and executing the mapped Python function!
            result = agent.run(query, llm_callback=llm_callback, max_retries=3, summarize_observation=True)
            print(f"Agent Execution Result: {result}\n")
            print("-" * 50)
        except Exception as e:
            print(f"Failed to execute agent: {e}\n")

if __name__ == "__main__":
    main()
