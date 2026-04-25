# Kiori

A minimalist Python agent library with as few dependencies as possible.

## Installation

```bash
# Basic installation
pip install kiori

# Installation with memory support
pip install "kiori[memory]"
```

## Quick Start

```python
from kiori.agent import KioriAgent
from kiori.models import Action, ActionExample

# Initialize agent
agent = KioriAgent()

# Define an action
def my_func():
    return "Action executed!"

action = Action(
    name="my_action",
    description="Does something cool",
    function_callable=my_func
)
agent.add_action(action)

# Add an example
example = ActionExample(
    user_prompt="Do something cool",
    expected_action_text="my_action()"
)
agent.add_example(example)

# Run agent
agent.run("Do something cool")
```

## Memory (LTM)

Kiori provides a `MilvusLTM` class for long-term memory using Milvus Lite.

```python
from kiori.memory import MilvusLTM
from kiori.models import ActionExample

# Initialize memory (auto-creates local Milvus Lite DB)
ltm = MilvusLTM()

example = ActionExample(
    user_prompt="Do something cool",
    expected_action_text="my_action()"
)

# Add examples to memory
ltm.add_examples([example])

# Search memory
results = ltm.search("Do something cool", top_k=1)
print(results)
```

## Philosophy
Kiori is built to be lightweight, easy to understand, and independent of bloated external packages.

## License
MIT
