from functools import lru_cache
from fastapi import Depends
from database import get_db_instance
from embedding_service import EmbeddingService
from rag_workflow import RAGWorkflow

@lru_cache()
def get_db():
    return get_db_instance()

@lru_cache()
def get_embedder():
    return EmbeddingService()

@lru_cache()
def get_rag_service(
    db = Depends(get_db), 
    embedder = Depends(get_embedder)
):
    return RAGWorkflow(db=db, embedder=embedder)