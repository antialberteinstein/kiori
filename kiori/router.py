from typing import Dict, List, Optional


class MarkovRouter:
    def __init__(
        self,
        transition_matrix: Optional[Dict[str, Dict[str, float]]] = None,
        all_actions: Optional[List[str]] = None
    ) -> None:
        self.transition_matrix = transition_matrix or {}
        self.all_actions = all_actions or []

    def get_probability(
        self, action_name: str, previous_action: Optional[str]
    ) -> float:
        n_actions = len(self.all_actions)
        default_prob = 1.0 / n_actions if n_actions > 0 else 0.0

        if not previous_action:
            return default_prob

        transitions = self.transition_matrix.get(previous_action)
        if transitions is not None:
            if action_name in transitions:
                return transitions[action_name]
            else:
                return 0.0

        return default_prob

    def calculate_combined_score(
        self,
        action_name: str,
        cosine_score: float,
        previous_action: Optional[str],
        alpha: float,
        beta: float
    ) -> float:
        prob = self.get_probability(action_name, previous_action)
        return alpha * cosine_score + beta * prob
