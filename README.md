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

## Execution Pipeline

Kiori provides a fully built-in execution pipeline that formats context, calls your LLM, parses the response, and executes the chosen Python function.

```python
from kiori.agent import KioriAgent
from kiori.models import Action

agent = KioriAgent()

def weather_action(location: str) -> str:
    return f"Weather in {location} is sunny."

agent.add_action(Action("get_weather", "Get weather for a location", weather_action))

# User provides their own LLM callback function
def my_llm_callback(prompt: str) -> str:
    # Call OpenAI, Claude, Gemini, etc.
    # The prompt instructs the LLM to output: [ACTION: name, ARGS: {...}]
    return '[ACTION: get_weather, ARGS: {"location": "Tokyo"}]'

# The agent automatically retrieves context, formats prompt, calls LLM, 
# executes the action, and saves the turn to Short-Term Memory!
result = agent.run("What's the weather in Tokyo?", llm_callback=my_llm_callback)
print(result) # "Weather in Tokyo is sunny."
```

## Routing & Probabilities (MarkovRouter)
Kiori allows combining semantic search (Cosine Similarity) with Markov Chain probabilities to resolve ambiguous prompts based on the conversation history.

```python
from kiori.router import MarkovRouter
from kiori.agent import KioriAgent

# Define transition matrix P(Action_B | Action_A)
transition_matrix = {
    "action_A": {"action_B": 0.9, "action_C": 0.1}
}

router = MarkovRouter(
    transition_matrix=transition_matrix,
    all_actions=["action_A", "action_B", "action_C"]
)

# Initialize agent with alpha (cosine weight) and beta (Markov weight)
agent = KioriAgent(ltm=ltm, router=router, alpha=0.5, beta=0.5)

# If the previous action was "action_A", the agent will scale up "action_B" 
# even if a new user prompt semantically looks slightly more like "action_C".
```

## Philosophy
Kiori is built to be lightweight, easy to understand, and independent of bloated external packages.

## License
MIT
