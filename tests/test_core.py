from unittest.mock import MagicMock
from kiori.agent import KioriAgent
from kiori.models import Action, ActionExample
from kiori.memory import ReplayBuffer


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


def test_context_shuffler() -> None:
    from kiori.agent import context_shuffler
    ex1 = ActionExample("p1", "a1")
    ex2 = ActionExample("p2", "a2")
    ex3 = ActionExample("p3", "a3")

    shuffled = context_shuffler([ex1, ex2, ex3])
    assert len(shuffled) == 3
    for ex in [ex1, ex2, ex3]:
        assert ex in shuffled


def test_format_prompt() -> None:
    from kiori.agent import format_prompt
    ex = ActionExample("p1", "a1")
    prompt = format_prompt("test", [ex])
    assert "User: p1" in prompt
    assert "Action: a1" in prompt
    assert "User: test\nAction:" in prompt


def test_run_pipeline() -> None:
    agent = KioriAgent()

    # Define an action
    def my_action(x: int) -> int:
        return x * 2

    agent.add_action(Action("test_action", "Multiply by 2", my_action))

    # Dummy LLM callback for immediate success
    def dummy_llm(prompt: str) -> str:
        return '[ACTION: test_action, ARGS: {"x": 10}]'

    # Run pipeline
    result = agent.run("Please multiply 10 by 2", dummy_llm)
    assert result == 20

    # Test Auto-Healing loop
    call_count = [0]

    def broken_then_fixed_llm(prompt: str) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            # Returns broken format with keyword 'ACTION:'
            return 'I will do [ACTION: test_action ARGS: x=15]'
        # Returns fixed format on retry
        return '[ACTION: test_action, ARGS: {"x": 15}]'

    result2 = agent.run("Multiply 15 by 2", broken_then_fixed_llm)
    assert result2 == 30
    assert call_count[0] == 2

    # Test Natural Chat
    def chat_llm(prompt: str) -> str:
        return 'Hello there!'

    result3 = agent.run("Hi", chat_llm)
    assert result3 == 'Hello there!'


def test_get_context_examples() -> None:
    # Setup mocks
    mock_ltm = MagicMock()
    ex_ltm = ActionExample("ltm_prompt", "ltm_action()")
    mock_ltm.search.return_value = [(ex_ltm, 0.9)]
    mock_ltm.scale_examples.return_value = [ex_ltm, ex_ltm]

    replay_buffer = ReplayBuffer()
    ex_replay = ActionExample("replay_prompt", "replay_action()")
    replay_buffer.update_buffer([ex_replay, ex_replay])

    agent = KioriAgent(ltm=mock_ltm, replay_buffer=replay_buffer)

    merged = agent.get_context_examples(
        "test prompt", threshold=0.5, max_copies=3, sample_n=1
    )

    assert len(merged) == 3
    assert merged.count(ex_ltm) == 2
    assert merged.count(ex_replay) == 1

    # Check mock called with right query
    mock_ltm.search.assert_called_once_with("test prompt", top_k=5)
