# Kiori

A minimalist Python agent library with as few dependencies as possible.

## Installation

```bash
pip install kiori
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

## Philosophy
Kiori is built to be lightweight, easy to understand, and independent of bloated external packages.

## License
MIT
