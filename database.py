import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dotenv import load_dotenv

# load env variables
load_dotenv()

class DB(ABC):
    """Abstract Base Class (Interface) for Storage operations."""
    @abstractmethod
    def upsert(self, doc_id: int, text: str, vector: List[float]) -> None:
        """
        Store a document with its embedding vector.
        
        Args:
            doc_id: Unique identifier for the document
            text: The document text content
            vector: Embedding vector for the document
        """
        pass

    @abstractmethod
    def search(self, query_vector: List[float], query_text: str, limit: int) -> List[str]:
        """
        Search for similar documents.
        
        Args:
            query_vector: Query embedding vector
            query_text: Query text to search
            limit: Maximum number of results to return
            
        Returns:
            List of matching document texts
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get storage status.
        
        Returns:
            Dictionary with storage metadata
        """
        pass

class InMemoryDB(DB):
    """In-memory storage implementation using a Python list."""
    def __init__(self):
        """Initialize empty storage."""
        self.memory_docs = []
    
    def upsert(self, doc_id: int, text: str, vector: List[float]) -> None:
        """Store document in memory."""
        self.memory_docs.append(text)
        return doc_id
    
    def search(self, query_vector: List[float], query_text: str, limit: int) -> List[str]:
        """Search documents by query text in documents."""
        results = []
        # basic keyword search
        for doc in self.memory_docs:
            if query_text.lower() in doc.lower():
                results.append(doc)
        
        # fallback if no keyword match, return the first document
        if not results and self.memory_docs:
            results = [self.memory_docs[0]]
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get in-memory storage status."""
        return {
            "qdrant_ready": False,
            "in_memory_docs_count": len(self.memory_docs)
        }

class QdrantDB(DB):
    """Implementation Qdrant Vector Database (Storage)."""
    def __init__(self, host: str):
        """Initialize Qdrant client and collection.
        
        Args:
            host: Qdrant server URL
        """
        # Qdrant setup (assumes local instance)
        self.client = QdrantClient(host)
        # recreate collection for demo purposes
        self.client.recreate_collection(
            collection_name="demo_collection",
            vectors_config=VectorParams(size=128, distance=Distance.COSINE)
        )

    def upsert(self, doc_id: int,  text: str, vector: List[float]) -> None:
        """Store document in Qdrant."""
        payload = {"text": text}
        
        # upload point to Qdrant using the provided UUID
        self.client.upsert(
            collection_name="demo_collection",
            points=[PointStruct(id=doc_id, vector=vector, payload=payload)]
        )
            
        return doc_id

    def search(self, query_vector: List[float], query_text: str, limit: int) -> List[str]:
        """Search for similar documents in Qdrant."""
        # semantic search using vector similarity
        hits = self.client.search(
            collection_name="demo_collection",
            query_vector=query_vector,
            limit=limit
        )
        return [hit.payload["text"] for hit in hits]

    def get_status(self) -> Dict[str, Any]:
        """Get Qdrant collection status."""
        return {
            "qdrant_ready": True,
            "in_memory_docs_count": 0 # not applicable for Qdrant mode so set to 0
        }
    
def get_db_instance() -> DB:
    """
    Function to initialize the database connection.
    Attempts to connect to Qdrant first; falls back to InMemoryDB on failure.
    """
    try:
        qdrant_host = os.getenv("QDRANT_HOST", "http://localhost:6333")
        # Initialize Qdrant
        db = QdrantDB(qdrant_host)
        return db
    except Exception as e:
        print("⚠️  Qdrant not available. Falling back to in-memory list.")
        return InMemoryDB()
    