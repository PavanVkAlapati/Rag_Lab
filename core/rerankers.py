# core/rerankers.py

from typing import List, Dict
import re

from openai import OpenAI

from config.settings import settings

_llm_client = None  # type: OpenAI

# Exposed options for the UI
AVAILABLE_RERANKERS = ["none", "llm", "hybrid_llm_distance"]


def get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(api_key=settings.openai_api_key)
    return _llm_client


def identity_rerank(chunks: List[Dict], question: str = None) -> List[Dict]:
    """
    No reranking. Use the order from the vector DB.
    """
    return chunks


def _llm_rank_indices(chunks: List[Dict], question: str) -> List[int]:
    """
    Ask the LLM to order the chunks by relevance to the question.
    Returns a list of indices (0..len(chunks)-1) in descending relevance.
    """
    if not chunks:
        return []

    client = get_llm_client()

    passages_text = []
    for i, ch in enumerate(chunks):
        text = ch.get("document", "")
        passages_text.append(f"[{i}] {text}")

    prompt = (
        "You are a passage reranker.\n"
        "Given a question and several passages, rank the passages by relevance.\n"
        "Return ONLY a comma-separated list of passage indices in descending order "
        "(most relevant first). No extra text.\n\n"
        f"Question:\n{question}\n\n"
        "Passages:\n" + "\n\n".join(passages_text)
    )

    completion = client.chat.completions.create(
        model=settings.llm_model_openai,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    content = completion.choices[0].message.content or ""
    # Extract integers
    indices = re.findall(r"\d+", content)
    ordered = []
    for idx_str in indices:
        try:
            idx = int(idx_str)
            if 0 <= idx < len(chunks) and idx not in ordered:
                ordered.append(idx)
        except ValueError:
            continue

    # Fallback: if parsing fails, just use original order
    if not ordered:
        ordered = list(range(len(chunks)))

    return ordered


def llm_rerank(chunks: List[Dict], question: str) -> List[Dict]:
    """
    Use the LLM to reorder chunks based purely on semantic relevance.
    """
    order = _llm_rank_indices(chunks, question)
    return [chunks[i] for i in order]


def hybrid_llm_distance_rerank(chunks: List[Dict], question: str) -> List[Dict]:
    """
    Hybrid reranker:
    - Vector DB already sorted by distance (most similar first).
    - LLM provides another ranking.
    - Combine rankings by simple Borda count: rank_vec + rank_llm (lower is better).
    """
    if not chunks:
        return []

    # base ranking from current order (vector distance)
    base_order = list(range(len(chunks)))
    base_rank = {idx: rank for rank, idx in enumerate(base_order)}

    # LLM ranking
    llm_order = _llm_rank_indices(chunks, question)
    llm_rank = {idx: rank for rank, idx in enumerate(llm_order)}

    # Combine ranks
    combined_scores = []
    for idx in range(len(chunks)):
        r_vec = base_rank.get(idx, len(chunks))
        r_llm = llm_rank.get(idx, len(chunks))
        combined = r_vec + r_llm
        combined_scores.append((combined, idx))

    combined_scores.sort(key=lambda x: x[0])
    ordered_indices = [idx for _, idx in combined_scores]

    return [chunks[i] for i in ordered_indices]


def rerank_chunks(
    question: str,
    chunks: List[Dict],
    strategy: str = "none",
) -> List[Dict]:
    """
    Unified entrypoint.
    strategy ∈ AVAILABLE_RERANKERS
    """
    strategy = strategy.lower()
    if strategy == "none":
        return identity_rerank(chunks, question)
    if strategy == "llm":
        return llm_rerank(chunks, question)
    if strategy == "hybrid_llm_distance":
        return hybrid_llm_distance_rerank(chunks, question)
    raise ValueError("Unknown reranker '%s'. Available: %s" % (strategy, AVAILABLE_RERANKERS))
