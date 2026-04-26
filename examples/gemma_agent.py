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

print("Loading LLM (unsloth/gemma-3-270m-it)... This might take a while.")
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
    model="unsloth/gemma-3-270m-it",
    torch_dtype=dtype,
    device=device
)

def llm_callback(prompt: str) -> str:
    """
    Callback function that connects Kiori's prompt to the Gemma model.
    """
    # Wrap Kiori's raw prompt inside the user message for Gemma's chat template
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    outputs = generator(messages, max_new_tokens=64)
    # The pipeline with messages returns the full conversation. We extract the last assistant response.
    generated_text = outputs[0]['generated_text'][-1]['content']
    
    # Optional: Print what the LLM actually generated for debugging
    print(f"\n[LLM Raw Output] -> {generated_text.strip()}\n")
    
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
        visible_files = [f for f in files if not f.startswith('.')]
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

    agent = KioriAgent(ltm=ltm, replay_buffer=replay_buffer)

    # Register Actions
    agent.add_action(Action("get_current_time", "Fetch the current time", get_current_time))
    agent.add_action(Action("calculate_expression", "Evaluate math expressions", calculate_expression))
    agent.add_action(Action("list_files", "List files in the directory", list_files))

    # Add Few-Shot Examples to Long-Term Memory
    # This teaches the LLM how to map user intents to the exact syntax we need
    examples = [
        ActionExample("What time is it right now?", '[ACTION: get_current_time, ARGS: {}]'),
        ActionExample("Tell me the current date and time", '[ACTION: get_current_time, ARGS: {}]'),
        
        ActionExample("Calculate 15 * 4", '[ACTION: calculate_expression, ARGS: {"expression": "15 * 4"}]'),
        ActionExample("What is 100 / 25 + 5?", '[ACTION: calculate_expression, ARGS: {"expression": "100 / 25 + 5"}]'),
        
        ActionExample("What files are in this folder?", '[ACTION: list_files, ARGS: {}]'),
        ActionExample("Show me the directory contents", '[ACTION: list_files, ARGS: {}]'),
    ]
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
            result = agent.run(query, llm_callback=llm_callback, max_retries=3)
            print(f"Agent Execution Result: {result}\n")
            print("-" * 50)
        except Exception as e:
            print(f"Failed to execute agent: {e}\n")

if __name__ == "__main__":
    main()
