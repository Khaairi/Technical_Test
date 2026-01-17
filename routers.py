import time
import uuid
from fastapi import APIRouter, HTTPException, Depends
import schemas as _schemas
from dependencies import get_db, get_embedder, get_rag_service
from database import DB
from embedding_service import EmbeddingService
from rag_workflow import RAGWorkflow

router = APIRouter()

@router.post("/ask")
def ask_question(
    req: _schemas.QuestionRequest,
    rag: RAGWorkflow = Depends(get_rag_service)
):
    """
    Endpoint to handle user questions using the RAG pipeline.

    Args:
        req: Question request
        rag: Injected RAG workflow instance
    """
    start = time.time()
    try:
        result = rag.process_question(req.question)
        return {
            "question": req.question,
            "answer": result["answer"],
            "context_used": result.get("context", []),
            "latency_sec": round(time.time() - start, 3)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/add")
def add_document(
    req: _schemas.DocumentRequest,
    db: DB = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedder)
):
    """
    Endpoint to ingest new documents to the knowledge base.
    
    Args:
        req: Document to add
        db: Injected document storage
        embedder: Injected embedding service
    """
    try:
        # generate a unique ID for the document
        doc_id = str(uuid.uuid4())
        # create vector embedding
        vector = embedder.embed(text=req.text)
        # save to storage 
        db.upsert(doc_id=doc_id, text=req.text, vector=vector)
        return {"id": doc_id, "status": "added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/status")
def status(
    db: DB = Depends(get_db),
    rag: RAGWorkflow = Depends(get_rag_service)
):
    """Get system status.
    
    Args:
        db: Injected document storage
        rag: Injected RAG workflow instance
        
    """
    db_status = db.get_status()
    chain_status = rag.get_status()
    return {
        "qdrant_ready": db_status["qdrant_ready"],
        "in_memory_docs_count": db_status["in_memory_docs_count"],
        "graph_ready": chain_status
    }