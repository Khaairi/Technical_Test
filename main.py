import time
from fastapi import FastAPI, HTTPException
import schemas as _schemas
from database import get_db_instance
from services import EmbeddingService, RAGService

db = get_db_instance()

embedder = EmbeddingService()

rag_service = RAGService(db=db, embedder=embedder)

app = FastAPI(title="Learning RAG Demo")

@app.post("/ask")
def ask_question(req: _schemas.QuestionRequest):
    start = time.time()
    try:
        result = rag_service.process_question(req.question)
        return {
            "question": req.question,
            "answer": result["answer"],
            "context_used": result.get("context", []),
            "latency_sec": round(time.time() - start, 3)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/add")
def add_document(req: _schemas.DocumentRequest):
    try:
        doc_id = rag_service.ingest_document(req.text)
        return {"id": doc_id, "status": "added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/status")
def status():
    db_status = db.get_status()
    chain_status = rag_service.get_status()
    return {
        "qdrant_ready": db_status["qdrant_ready"],
        "in_memory_docs_count": db_status["in_memory_docs_count"],
        "graph_ready": chain_status
    }