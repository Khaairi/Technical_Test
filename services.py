import random
from langgraph.graph import StateGraph, END
from typing import Dict, List
from database import DB
import uuid

class EmbeddingService:
    def fake_embed(self, text: str) -> List[float]:
        # Seed based on input so it's "deterministic"
        random.seed(abs(hash(text)) % 10000)
        return [random.random() for _ in range(128)] # Small vector for demo
    
class RAGService:
    def __init__(self, db: DB, embedder: EmbeddingService):
        self.db = db 
        self.embedder = embedder
        self.chain = self._build_graph()

    def _retrieve_node(self, state: Dict):
        query = state["question"]
        
        vector = self.embedder.fake_embed(text=query)
        
        docs = self.db.search(query_vector=vector, query_text=query, limit=2)
        
        state["context"] = docs
        return state

    def _answer_node(self, state: Dict):
        ctx = state["context"]
        if ctx:
            answer = f"I found this: '{ctx[0][:100]}...'"
        else:
            answer = "Sorry, I don't know."
        state["answer"] = answer
        return state

    def _build_graph(self):
        workflow = StateGraph(dict)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("answer", self._answer_node)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", END)
        return workflow.compile()

    def process_question(self, question: str) -> Dict:
        return self.chain.invoke({"question": question})

    def ingest_document(self, text: str) -> str:
        doc_id = str(uuid.uuid4())
        vector = self.embedder.fake_embed(text=text)
        return self.db.upsert(doc_id=doc_id, text=text, vector=vector)
    
    def get_status(self) -> bool:
        return self.chain is not None