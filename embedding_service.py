import random
from typing import List

class EmbeddingService:
    """Service responsible for converting text into vector representations."""
    def embed(self, text: str) -> List[float]:
        """
        Generate a deterministic embedding vector for the given text.
        
        Args:
            text: Input text to embed
            
        Returns:
            A list of floats representing the embedding vector
        """
        # Seed based on input so it's "deterministic"
        random.seed(abs(hash(text)) % 10000)
        return [random.random() for _ in range(128)] # Small vector for demo