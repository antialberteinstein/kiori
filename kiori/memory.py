import random
from typing import List, Tuple

from .models import ActionExample


class MilvusLTM:
    def __init__(
        self,
        db_path: str = "./kiori_memory.db",
        collection_name: str = "kiori_examples",
        model_name: str = "all-MiniLM-L6-v2"
    ) -> None:
        try:
            from pymilvus import MilvusClient
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("MilvusLTM requires 'pymilvus' and 'sentence-transformers'. Install with: pip install \"kiori[memory]\"")
            
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

    def add_examples(self, examples: List[ActionExample], similarity_threshold: float = 0.95) -> None:
        """Thêm danh sách ví dụ vào bộ nhớ dài hạn, bỏ qua các ví dụ quá giống nhau."""
        if not examples:
            return
            
        # Đảm bảo collection tồn tại trước khi insert
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dim,
                metric_type="COSINE",
                id_type="int",
                auto_id=True
            )

        to_insert = []
        for ex in examples:
            # Kiểm tra xem ví dụ này đã tồn tại chưa (Similarity check)
            existing = self.search(ex.user_prompt, top_k=1)
            if existing:
                _, score = existing[0]
                if score >= similarity_threshold:
                    # Ví dụ này đã tồn tại hoặc rất giống, bỏ qua
                    continue
            
            to_insert.append(ex)

        if not to_insert:
            return

        prompts = [ex.user_prompt for ex in to_insert]
        embeddings = self.model.encode(prompts, convert_to_numpy=True).tolist()
        
        data = [
            {
                "user_prompt": ex.user_prompt,
                "expected_action_text": ex.expected_action_text,
                "vector": embeddings[i]
            }
            for i, ex in enumerate(to_insert)
        ]

        self.client.insert(collection_name=self.collection_name, data=data)

    def clear(self) -> None:
        """Xóa toàn bộ dữ liệu trong collection hiện tại."""
        # Cách nhanh nhất là drop và tạo lại hoặc xóa bằng filter
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        # Tự động khởi tạo lại khi insert

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
        """Lọc và nhân bản ví dụ dựa trên điểm số. Đảm bảo mỗi ví dụ hợp lệ xuất hiện ít nhất 1 lần."""
        scaled_examples = []
        for ex, score in examples_with_scores:
            if score >= threshold:
                # Nếu max_copies = 1, chúng ta chỉ lấy duy nhất 1 bản sao
                if max_copies <= 1:
                    scaled_examples.append(ex)
                else:
                    # Nhân bản dựa trên độ tương đồng (score)
                    ratio = min(1.0, max(0.0, score))
                    copies = max(1, int(ratio * max_copies))
                    scaled_examples.extend([ex] * copies)
        return scaled_examples


class ReplayBuffer:

    NO_REPLAY_BUFFER = None

    def __init__(self) -> None:
        self.buffer: List[ActionExample] = []

    def update_buffer(self, new_examples: List[ActionExample]) -> None:
        self.buffer = list(new_examples)

    def sample_buffer(self, n: int) -> List[ActionExample]:
        if n >= len(self.buffer):
            return list(self.buffer)
        return random.sample(self.buffer, n)
