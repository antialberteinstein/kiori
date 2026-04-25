from typing import List, Tuple
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from .models import ActionExample


class MilvusLTM:
    def __init__(
        self,
        db_path: str = "./kiori_memory.db",
        collection_name: str = "kiori_examples",
        model_name: str = "all-MiniLM-L6-v2"
    ) -> None:
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

        if not self.client.has_collection(
            collection_name=self.collection_name
        ):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dim,
                metric_type="COSINE",
                id_type="int",
                auto_id=True
            )

    def add_examples(self, examples: List[ActionExample]) -> None:
        if not examples:
            return

        prompts = [ex.user_prompt for ex in examples]
        embeddings = self.model.encode(prompts, convert_to_numpy=True).tolist()

        data = []
        for ex, emb in zip(examples, embeddings):
            data.append({
                "vector": emb,
                "user_prompt": ex.user_prompt,
                "expected_action_text": ex.expected_action_text
            })

        self.client.insert(collection_name=self.collection_name, data=data)

    def search(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[ActionExample, float]]:
        emb = self.model.encode([query], convert_to_numpy=True).tolist()[0]
        results = self.client.search(
            collection_name=self.collection_name,
            data=[emb],
            limit=top_k,
            output_fields=["user_prompt", "expected_action_text"]
        )

        examples_with_scores = []
        if results and results[0]:
            for hit in results[0]:
                score = hit.get("distance", 0.0)
                entity = hit.get("entity", {})
                ex = ActionExample(
                    user_prompt=entity.get("user_prompt", ""),
                    expected_action_text=entity.get("expected_action_text", "")
                )
                examples_with_scores.append((ex, score))
        return examples_with_scores

    def scale_examples(
        self,
        examples_with_scores: List[Tuple[ActionExample, float]],
        threshold: float,
        max_copies: int
    ) -> List[ActionExample]:
        scaled_examples = []
        for ex, score in examples_with_scores:
            if score > threshold:
                # Duplicate example proportional to its score
                # Ensuring it generates between 1 and max_copies
                ratio = min(1.0, max(0.0, score))
                copies = max(1, int(ratio * max_copies))
                scaled_examples.extend([ex] * copies)
        return scaled_examples
