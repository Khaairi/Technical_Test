from functools import lru_cache
from fastapi import Depends
from database import get_db_instance
from embedding_service import EmbeddingService
from rag_workflow import RAGWorkflow

@lru_cache()
def get_db():
    """
    Provides a singleton instance of the Database (Storage).
    lru_cache ensures the connection is established only once.
    """
    return get_db_instance()

@lru_cache()
def get_embedder():
    """Provides a singleton instance of the EmbeddingService."""
    return EmbeddingService()

@lru_cache()
def get_rag_service(
    db = Depends(get_db), 
    embedder = Depends(get_embedder)
):
    """Constructs the RAGWorkflow by injecting the required DB and Embedder instances."""
    return RAGWorkflow(db=db, embedder=embedder)