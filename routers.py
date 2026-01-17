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
    try:
        doc_id = str(uuid.uuid4())
        vector = embedder.embed(text=req.text)
        saved_id = db.upsert(doc_id=doc_id, text=req.text, vector=vector)
        return {"id": saved_id, "status": "added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/status")
def status(
    db: DB = Depends(get_db),
    rag: RAGWorkflow = Depends(get_rag_service)
):
    db_status = db.get_status()
    chain_status = rag.get_status()
    return {
        "qdrant_ready": db_status["qdrant_ready"],
        "in_memory_docs_count": db_status["in_memory_docs_count"],
        "graph_ready": chain_status
    }