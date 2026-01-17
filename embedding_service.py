import random
from typing import List

class EmbeddingService:
    def embed(self, text: str) -> List[float]:
        # Seed based on input so it's "deterministic"
        random.seed(abs(hash(text)) % 10000)
        return [random.random() for _ in range(128)] # Small vector for demo