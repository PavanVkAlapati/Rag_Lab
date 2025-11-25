# metrics/eval_metrics.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from core.rag_pipeline import answer_question
from metrics.loggig_excel import log_to_excel


def _load_eval_set(path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation data from a JSON file.
    Expected format:
    [
        {
            "id": 1,
            "question": "What is ... ?",
            "expected_answer": "Some reference answer"
        },
        ...
    ]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Eval file must contain a JSON list of objects")

    return data


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _overlap_score(expected: str, predicted: str) -> float:
    """
    Very simple token overlap score between 0 and 1.
    |intersection(tokens)| / |union(tokens)|
    """
    exp_tokens = set(_tokenize(expected))
    pred_tokens = set(_tokenize(predicted))

    if not exp_tokens and not pred_tokens:
        return 1.0
    if not exp_tokens or not pred_tokens:
        return 0.0

    inter = exp_tokens.intersection(pred_tokens)
    union = exp_tokens.union(pred_tokens)
    return float(len(inter)) / float(len(union))


def evaluate_rag(
    eval_path: str = "data/eval_set.json",
    collection_name: str = "demo",
    k: int = 4,
    excel_path: str = "logs/rag_eval_log.xlsx",
) -> Dict[str, Any]:
    """
    Run the vanilla RAG pipeline on an eval set and compute simple metrics.
    Returns a dict with aggregate statistics.
    Also logs per-example results to an Excel file.
    """
    eval_items = _load_eval_set(eval_path)

    results: List[Dict[str, Any]] = []
    contains_matches = 0
    overlap_scores: List[float] = []

    for item in eval_items:
        q = item.get("question", "")
        expected = item.get("expected_answer", "")
        sample_id = item.get("id")

        rag_result = answer_question(
            question=q,
            collection_name=collection_name,
            k=k,
        )

        answer = rag_result["answer"] or ""

        # simple metrics
        contains = expected.lower() in answer.lower() if expected else False
        overlap = _overlap_score(expected, answer) if expected else 0.0

        if contains:
            contains_matches += 1
        overlap_scores.append(overlap)

        row = {
            "id": sample_id,
            "question": q,
            "expected_answer": expected,
            "model_answer": answer,
            "contains_match": contains,
            "overlap_score": overlap,
        }
        results.append(row)

    # log to Excel
    if results:
        log_to_excel(results, excel_path=excel_path)

    n = len(eval_items) if eval_items else 0
    contains_accuracy = (contains_matches / n) if n > 0 else 0.0
    avg_overlap = (sum(overlap_scores) / len(overlap_scores)) if overlap_scores else 0.0

    summary = {
        "n_samples": n,
        "contains_accuracy": contains_accuracy,
        "avg_overlap_score": avg_overlap,
    }

    print("=== RAG Evaluation Summary ===")
    print(f"Samples:           {n}")
    print(f"Contains accuracy: {contains_accuracy:.3f}")
    print(f"Avg overlap score: {avg_overlap:.3f}")

    return summary


if __name__ == "__main__":
    # Default command-line entry point
    evaluate_rag()
