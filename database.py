import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Abstract class
class DB(ABC):
    @abstractmethod
    def upsert(self, doc_id: int, text: str, vector: List[float]) -> str:
        pass

    @abstractmethod
    def search(self, query_vector: List[float], query_text: str, limit: int) -> List[str]:
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        pass

class InMemoryDB(DB):
    def __init__(self):
        self.memory_docs = []
    
    def upsert(self, doc_id: int, text: str, vector: List[float]) -> str:
        self.memory_docs.append(text)
        return doc_id
    
    def search(self, query_vector: List[float], query_text: str, limit: int) -> List[str]:
        results = []
        for doc in self.memory_docs:
            if query_text.lower() in doc.lower():
                results.append(doc)
        
        if not results and self.memory_docs:
            results = [self.memory_docs[0]]
        return results
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "qdrant_ready": False,
            "in_memory_docs_count": len(self.memory_docs)
        }

class QdrantDB(DB):
    def __init__(self, host: str):
        # Qdrant setup (assumes local instance)
        self.client = QdrantClient(host)
        self.client.recreate_collection(
            collection_name="demo_collection",
            vectors_config=VectorParams(size=128, distance=Distance.COSINE)
        )

    def upsert(self, doc_id: int,  text: str, vector: List[float]) -> str:
        payload = {"text": text}
        
        self.client.upsert(
            collection_name="demo_collection",
            points=[PointStruct(id=doc_id, vector=vector, payload=payload)]
        )
            
        return doc_id

    def search(self, query_vector: List[float], query_text: str, limit: int) -> List[str]:
        hits = self.client.search(
            collection_name="demo_collection",
            query_vector=query_vector,
            limit=limit
        )
        return [hit.payload["text"] for hit in hits]

    def get_status(self) -> Dict[str, Any]:
        return {
            "qdrant_ready": True,
            "in_memory_docs_count": 0
        }
    
def get_db_instance() -> DB:
    try:
        qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
        db = QdrantDB(qdrant_host)
        return db
    except Exception as e:
        print("⚠️  Qdrant not available. Falling back to in-memory list.")
        return InMemoryDB()
    