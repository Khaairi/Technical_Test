from pydantic import BaseModel

class QuestionRequest(BaseModel):
    """Schema for the question input payload."""
    question: str

class DocumentRequest(BaseModel):
    """Schema for the document ingestion payload."""
    text: str