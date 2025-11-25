from fastapi import FastAPI
from pydantic import BaseModel

from core.rag_pipeline import answer_question

app = FastAPI(title="Vanilla RAG API", version="0.1.0")


class QueryRequest(BaseModel):
    question: str
    collection_name: str = "demo"
    k: int = 4


class QueryResponse(BaseModel):
    question: str
    answer: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    result = answer_question(
        question=payload.question,
        collection_name=payload.collection_name,
        k=payload.k,
    )
    return QueryResponse(question=result["question"], answer=result["answer"])
