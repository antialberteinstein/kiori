from typing import Any
from kiori.agent import KioriAgent
from kiori.models import Action, ActionExample


def dummy_func() -> str:
    return "dummy"


def test_agent_initialization() -> None:
    agent = KioriAgent()
    assert len(agent.actions) == 0
    assert len(agent.examples) == 0


def test_add_action() -> None:
    agent = KioriAgent()
    action = Action(
        name="test_action",
        description="A test action",
        function_callable=dummy_func
    )
    agent.add_action(action)
    assert len(agent.actions) == 1
    assert agent.actions[0].name == "test_action"


def test_add_example() -> None:
    agent = KioriAgent()
    example = ActionExample(
        user_prompt="Do the test",
        expected_action_text="test_action()"
    )
    agent.add_example(example)
    assert len(agent.examples) == 1
    assert agent.examples[0].user_prompt == "Do the test"


def test_run(capsys: Any) -> None:
    agent = KioriAgent()
    agent.run("Hello world")
    captured = capsys.readouterr()
    assert "Prompt received: Hello world\n" == captured.out
