# Kiori 🧠

[![PyPI version](https://badge.fury.io/py/kiori.svg)](https://badge.fury.io/py/kiori)
[![Python versions](https://img.shields.io/pypi/pyversions/kiori.svg)](https://pypi.org/project/kiori/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Kiori** is a minimalist, highly extensible Python framework for building SLM/LLM-based agents. True to its philosophy, it eschews bloated dependencies in favor of a clean, composable architecture. 

In version 1.1.0, Kiori embraces the inherent reasoning power of modern Small Language Models (SLMs) and LLMs by relying entirely on **Long-Term Memory (LTM)** via Cosine Similarity and **Short-Term Memory (STM)** via a Replay Buffer. This dynamic memory injection provides agents with robust contextual awareness without artificial routing constraints.

---

## Architecture Overview

Kiori seamlessly merges different memory paradigms before sending prompts to your model:

1. **Long-Term Memory (LTM) & Weighted Pattern Matcher**: Powered by Milvus Lite, the LTM stores past interactions and few-shot examples as vector embeddings. When a new prompt arrives, Kiori performs semantic search (Cosine Similarity). High-confidence matches are dynamically duplicated (scaled) to heavily influence the model's behavior via the Weighted Pattern Matcher mechanism.
2. **Short-Term Memory (Replay Buffer)**: Retains the immediate preceding turns of a conversation, ensuring the model is acutely aware of recent context and can reason about the ongoing dialogue state natively.
3. **Execution Pipeline**: Automatically parses the model's string output, extracts action signatures and arguments, and executes the mapped Python code.

---

## Installation

```bash
# Core minimal installation
pip install kiori

# Install with memory module dependencies (pymilvus & sentence-transformers)
pip install "kiori[memory]"
```

## Quick Start: End-to-End Workflow

Below is a complete example demonstrating how to initialize an Agent, set up its Memory, add Actions, and execute a user prompt.

```python
from kiori.agent import KioriAgent
from kiori.models import Action, ActionExample
from kiori.memory import MilvusLTM, ReplayBuffer

# 1. Setup Memory Modules
ltm = MilvusLTM()  # Automatically initializes a local Milvus Lite vector DB
replay_buffer = ReplayBuffer()

# 2. Add some prior knowledge (Few-shot examples) to LTM
example = ActionExample(
    user_prompt="Check the server status",
    expected_action_text="[ACTION: get_status, ARGS: {}]"
)
ltm.add_examples([example])

# 3. Initialize the Kiori Agent
agent = KioriAgent(
    ltm=ltm, 
    replay_buffer=replay_buffer
)

# 4. Define and Register Actions (Python Callables)
def get_status() -> str:
    return "Server is running smoothly."

def restart_server() -> str:
    return "Restarting server..."

agent.add_action(Action("get_status", "Fetches current server status", get_status))
agent.add_action(Action("restart_server", "Restarts the server", restart_server))

# 5. Provide an LLM Callback
# This function connects Kiori to your LLM of choice (OpenAI, Anthropic, Gemini, etc.)
def my_llm_callback(prompt: str) -> str:
    # Example: Send the prompt to an LLM API.
    # The prompt natively instructs the LLM to output [ACTION: name, ARGS: {...}]
    
    # Simulating LLM response based on the prompt for demonstration:
    return '[ACTION: get_status, ARGS: {}]'

# 6. Execute the Pipeline
# The Agent retrieves LTM, samples STM, shuffles context, 
# prompts the LLM, parses the response, executes the action, and updates the buffer!
result = agent.run("Is the server okay?", llm_callback=my_llm_callback)

print(result) # Output: "Server is running smoothly."
```

## Philosophy

Kiori is built to be lightweight, easy to understand, and independent of bloated external packages. By keeping the core architecture minimalistic, developers have full freedom to compose and customize the AI execution flow without being tied down by rigidly opinionated design patterns.
