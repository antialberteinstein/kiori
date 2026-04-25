from unittest.mock import MagicMock, patch
import numpy as np
from kiori.models import ActionExample
from kiori.memory import MilvusLTM, ReplayBuffer


@patch('kiori.memory.SentenceTransformer')
@patch('kiori.memory.MilvusClient')
def test_milvus_ltm_scale(mock_milvus: MagicMock, mock_st: MagicMock) -> None:
    # Setup mocks
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_st.return_value = mock_model

    mock_client = MagicMock()
    mock_client.has_collection.return_value = True
    mock_milvus.return_value = mock_client

    ltm = MilvusLTM()

    # Test scale logic
    ex1 = ActionExample("p1", "a1")
    ex2 = ActionExample("p2", "a2")
    ex3 = ActionExample("p3", "a3")

    examples_with_scores = [
        (ex1, 0.9),
        (ex2, 0.5),
        (ex3, 0.2)
    ]

    # threshold 0.4, max_copies 5
    # ex1: score 0.9 > 0.4. copies = max(1, int(0.9 * 5)) = 4
    # ex2: score 0.5 > 0.4. copies = max(1, int(0.5 * 5)) = 2
    # ex3: score 0.2 < 0.4. filtered out.
    scaled = ltm.scale_examples(
        examples_with_scores,
        threshold=0.4,
        max_copies=5
    )

    assert len(scaled) == 6
    assert scaled.count(ex1) == 4
    assert scaled.count(ex2) == 2
    assert scaled.count(ex3) == 0


@patch('kiori.memory.SentenceTransformer')
@patch('kiori.memory.MilvusClient')
def test_milvus_search(mock_milvus: MagicMock, mock_st: MagicMock) -> None:
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])
    mock_model.get_sentence_embedding_dimension.return_value = 2
    mock_st.return_value = mock_model

    mock_client = MagicMock()
    mock_client.search.return_value = [
        [
            {
                "distance": 0.8,
                "entity": {
                    "user_prompt": "hello",
                    "expected_action_text": "greet()"
                }
            }
        ]
    ]
    mock_milvus.return_value = mock_client

    ltm = MilvusLTM()
    results = ltm.search("hi")

    assert len(results) == 1
    assert results[0][1] == 0.8
    assert results[0][0].user_prompt == "hello"
    assert results[0][0].expected_action_text == "greet()"


def test_replay_buffer() -> None:
    rb = ReplayBuffer()
    assert len(rb.buffer) == 0

    ex1 = ActionExample("p1", "a1")
    ex2 = ActionExample("p2", "a2")
    ex3 = ActionExample("p3", "a3")

    rb.update_buffer([ex1, ex2, ex3])
    assert len(rb.buffer) == 3

    # Sample more than exists
    sampled = rb.sample_buffer(5)
    assert len(sampled) == 3

    # Sample less
    sampled2 = rb.sample_buffer(2)
    assert len(sampled2) == 2
    assert sampled2[0] in [ex1, ex2, ex3]
