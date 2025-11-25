# core/rag_pipeline.py

from typing import Dict, List

from openai import OpenAI

from config.settings import settings
from core.retrievers import retrieve
from core.rerankers import rerank_chunks


_llm_client = None  # type: OpenAI


def get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(api_key=settings.openai_api_key)
    return _llm_client


def build_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a single context string.
    """
    parts: List[str] = []
    for i, doc in enumerate(chunks, start=1):
        meta = doc.get("metadata") or {}
        source = meta.get("source", "unknown")
        header = f"[Chunk {i} | Source: {source}]"
        parts.append(f"{header}\n{doc['document']}")
    return "\n\n".join(parts)


def answer_question(
    question: str,
    collection_name: str = "demo",
    k: int = 4,
    db_type: str = "chroma",
    embedding_backend: str = None,
    reranker: str = "none",
) -> Dict:
    """
    Full vanilla RAG pipeline:
    - Retrieve docs from chosen vector DB
    - Rerank using selected strategy
    - Build context
    - Ask LLM
    """
    # Retrieve top-k (vector only)
    retrieved = retrieve(
        question,
        collection_name=collection_name,
        k=k,
        db_type=db_type,
        embedding_backend=embedding_backend,
    )

    # Rerank
    reranked = rerank_chunks(
        question=question,
        chunks=retrieved,
        strategy=reranker,
    )

    context = build_context(reranked)

    client = get_llm_client()
    system_prompt = (
        "You are a retrieval-augmented QA assistant.\n"
        "Use ONLY the provided context to answer the question.\n"
        "If the answer is not in the context, say you don't know."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context}",
        },
    ]

    completion = client.chat.completions.create(
        model=settings.llm_model_openai,
        messages=messages,
        temperature=0.1,
    )

    answer = completion.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "retrieved": reranked,
    }
