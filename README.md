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

## Memory (LTM & STM)

Kiori provides memory modules for both Long-Term Memory (LTM) and Short-Term Memory (STM).

### Long-Term Memory (MilvusLTM)

Kiori provides a `MilvusLTM` class using Milvus Lite.

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
```

### Short-Term Memory (ReplayBuffer)

`ReplayBuffer` stores examples from the immediate previous conversation turn.

```python
from kiori.memory import ReplayBuffer

replay_buffer = ReplayBuffer()
replay_buffer.update_buffer([example])
```

### Context Integration

`KioriAgent` automatically merges contexts from both memory sources.

```python
from kiori.agent import KioriAgent

agent = KioriAgent(ltm=ltm, replay_buffer=replay_buffer)

# Automatically searches LTM and samples STM to generate context examples
ctx = agent.get_context_examples("User's new prompt")
print(ctx)
```

## Philosophy
Kiori is built to be lightweight, easy to understand, and independent of bloated external packages.

## License
MIT
