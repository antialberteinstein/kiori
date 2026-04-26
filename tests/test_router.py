from kiori.models import ActionExample
from kiori.router import MarkovRouter
from kiori.agent import KioriAgent
from unittest.mock import MagicMock


def test_markov_router_combined_score() -> None:
    tm = {
        "action_A": {"action_B": 0.9, "action_C": 0.1}
    }
    router = MarkovRouter(
        transition_matrix=tm,
        all_actions=["action_A", "action_B", "action_C"]
    )

    # action_B has high Markov prob (0.9), low cosine (0.2)
    score_b = router.calculate_combined_score(
        "action_B", 0.2, "action_A", alpha=0.5, beta=0.5
    )
    # 0.5 * 0.2 + 0.5 * 0.9 = 0.1 + 0.45 = 0.55
    assert abs(score_b - 0.55) < 1e-5

    # action_C has high cosine (0.8), low Markov prob (0.1)
    score_c = router.calculate_combined_score(
        "action_C", 0.8, "action_A", alpha=0.5, beta=0.5
    )
    # 0.5 * 0.8 + 0.5 * 0.1 = 0.4 + 0.05 = 0.45
    assert abs(score_c - 0.45) < 1e-5

    # Prove B scores higher than C
    assert score_b > score_c


def test_agent_integration_with_router() -> None:
    mock_ltm = MagicMock()
    ex_b = ActionExample("do b", "action_B()")
    ex_c = ActionExample("do c", "action_C()")

    # Milvus returns C with higher cosine score than B
    mock_ltm.search.return_value = [(ex_b, 0.2), (ex_c, 0.8)]

    # Mock scale_examples to return a list proportional to the score passed
    def mock_scale(examples_with_scores, threshold, max_copies):
        res = []
        for ex, score in examples_with_scores:
            if score > threshold:
                copies = max(1, int(score * max_copies))
                res.extend([ex] * copies)
        return res

    mock_ltm.scale_examples.side_effect = mock_scale

    tm = {
        "action_A": {"action_B": 0.9, "action_C": 0.1}
    }
    router = MarkovRouter(
        transition_matrix=tm,
        all_actions=["action_A", "action_B", "action_C"]
    )

    agent = KioriAgent(ltm=mock_ltm, router=router)
    agent.previous_action = "action_A"

    # Get context examples with threshold 0.5, max_copies 10
    ctx = agent.get_context_examples("test", threshold=0.5, max_copies=10)

    # score_B = 0.55 -> copies = int(0.55 * 10) = 5
    # score_C = 0.45 -> filtered out (<= 0.5)

    assert ctx.count(ex_b) == 5
    assert ctx.count(ex_c) == 0
