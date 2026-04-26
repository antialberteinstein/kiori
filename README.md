# Kiori 🧠

[![PyPI version](https://badge.fury.io/py/kiori.svg)](https://badge.fury.io/py/kiori)
[![Python versions](https://img.shields.io/pypi/pyversions/kiori.svg)](https://pypi.org/project/kiori/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Kiori** is a minimalist, highly extensible Python framework for building SLM/LLM-based agents. True to its philosophy, it eschews bloated dependencies in favor of a clean, composable architecture. 

In version 1.2.0, Kiori embraces **Zero-Configuration** and **Auto-Healing**. It relies on the reasoning power of modern Small Language Models (SLMs) and provides built-in mechanisms to automatically correct model syntax errors.

---

## Architecture Overview

Kiori seamlessly merges different paradigms before sending prompts to your model:

1. **Long-Term Memory (LTM) & Weighted Pattern Matcher**: Powered by Milvus Lite, the LTM stores past interactions and few-shot examples as vector embeddings. When a new prompt arrives, Kiori performs semantic search (Cosine Similarity). High-confidence matches are dynamically duplicated (scaled) to heavily influence the model's behavior.
2. **Short-Term Memory (Replay Buffer)**: Retains the immediate preceding turns of a conversation, ensuring the model is acutely aware of recent context. *Note: Kiori intelligently filters out broken interactions, ensuring your SLM only learns from perfect syntax.*
3. **Smart Parser (`KioriParser`)**: Evaluates the LLM's output and categorizes it into three distinct states:
   - `SUCCESS`: The LLM output perfectly matches the expected `[ACTION: name, ARGS: {...}]` format.
   - `NATURAL_CHAT`: The LLM is just conversing naturally with the user.
   - `BROKEN_FORMAT`: The LLM attempted an action but broke the JSON or bracket syntax.
4. **Auto-Healing Loop**: If the LLM generates a `BROKEN_FORMAT`, Kiori automatically increments a retry counter, appends a system observation (`[System Observation: Text của bạn bị sai định dạng...]`), and prompts the LLM to correct its own mistake instantly.

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

# 3. Initialize the Kiori Agent (Zero-Configuration!)
agent = KioriAgent(ltm=ltm, replay_buffer=replay_buffer)

# 4. Define and Register Actions (Python Callables)
def get_status() -> str:
    return "Server is running smoothly."

agent.add_action(Action("get_status", "Fetches current server status", get_status))

# 5. Provide an LLM Callback
def my_llm_callback(prompt: str) -> str:
    # Simulating a broken LLM response for demonstration:
    return 'I will run the command: [ACTION: get_status ARGS: {}' # Missing comma and closing bracket!

# 6. Execute the Pipeline
# Kiori will automatically detect the BROKEN_FORMAT, append a correction prompt, 
# and recall `my_llm_callback` until the LLM outputs a valid SUCCESS format or hits max_retries!
result = agent.run("Is the server okay?", llm_callback=my_llm_callback, max_retries=3)

print(result)
```

## Philosophy

Kiori is built to be lightweight, easy to understand, and independent of bloated external packages. By keeping the core architecture minimalistic, developers have full freedom to compose and customize the AI execution flow without being tied down by rigidly opinionated design patterns.
