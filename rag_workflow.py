from langgraph.graph import StateGraph, END
from typing import Dict, Any
from database import DB
from embedding_service import EmbeddingService
    
class RAGWorkflow:
    """Orchestrates the Retrieval-Augmented Generation (RAG) flow using LangGraph."""
    def __init__(self, db: DB, embedder: EmbeddingService):
        """Initialize the RAG workflow.
        
        Args:
            db: Storage for documents
            embedder: Service for generating embeddings
        """
        self.db = db 
        self.embedder = embedder
        # build the graph upon initialization
        self.chain = self._build_graph()

    def _retrieve_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant documents based on the question.
        
        Args:
            state: Current workflow state containing 'question'
            
        Returns:
            Updated state with 'context' field
        """
        query = state["question"]
        
        # convert query to vector
        vector = self.embedder.embed(text=query)
        
        # search storage
        docs = self.db.search(query_vector=vector, query_text=query, limit=2)
        
        # update state with found context
        state["context"] = docs
        return state

    def _answer_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer based on retrieved context.
        
        Args:
            state: Current workflow state containing 'context'
            
        Returns:
            Updated state with 'answer' field
        """
        # dummy generation logic
        ctx = state["context"]
        if ctx:
            answer = f"I found this: '{ctx[0][:100]}...'"
        else:
            answer = "Sorry, I don't know."
        # update state with answer
        state["answer"] = answer
        return state

    def _build_graph(self):
        """Construct the LangGraph workflow."""
        workflow = StateGraph(dict)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("answer", self._answer_node)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "answer")
        workflow.add_edge("answer", END)
        return workflow.compile()

    def process_question(self, question: str) -> Dict[str, Any]:
        """Process full RAG workflow for a question.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer and context
        """
        return self.chain.invoke({"question": question})
    
    def get_status(self) -> bool:
        """Checks if the LangGraph chain is initialized."""
        return self.chain is not None