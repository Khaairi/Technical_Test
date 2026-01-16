from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from abc import ABC, abstractmethod
from typing import List, Dict, Any

# Abstract class
class DB(ABC):
    @abstractmethod
    def upsert(self, text: str, vector: List[float]) -> int:
        pass

    @abstractmethod
    def search(self, query_vector: List[float], query_text: str, limit: int) -> List[str]:
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        pass

class QdrantDB(DB):
    def __init__(self, host: str = "http://localhost:6333"):
        self.memory_docs = [] # Super basic in-memory "storage" fallback
        self.using_qdrant = False
        
        # Qdrant setup (assumes local instance)
        try:
            self.client = QdrantClient(host)
            self.client.recreate_collection(
                collection_name="demo_collection",
                vectors_config=VectorParams(size=128, distance=Distance.COSINE)
            )
            self.using_qdrant = True
        except Exception:
            print("⚠️  Qdrant not available. Falling back to in-memory list.")
            self.using_qdrant = False

    def upsert(self, text: str, vector: List[float]) -> int:
        doc_id = len(self.memory_docs)
        payload = {"text": text}
        
        if self.using_qdrant:
            self.client.upsert(
                collection_name="demo_collection",
                points=[PointStruct(id=doc_id, vector=vector, payload=payload)]
            )
        else:
            self.memory_docs.append(text)
            
        return doc_id

    def search(self, query_vector: List[float], query_text: str, limit: int) -> List[str]:
        results = []
        
        if self.using_qdrant:
            hits = self.client.search(
                collection_name="demo_collection",
                query_vector=query_vector,
                limit=limit
            )
            results = [hit.payload["text"] for hit in hits]
        else:
            for doc in self.memory_docs:
                if query_text.lower() in doc.lower():
                    results.append(doc)
            if not results and self.memory_docs:
                results = [self.memory_docs[0]] # Just grab first
                
        return results

    def get_status(self) -> Dict[str, Any]:
        return {
            "qdrant_ready": self.using_qdrant,
            "in_memory_docs_count": len(self.memory_docs)
        }