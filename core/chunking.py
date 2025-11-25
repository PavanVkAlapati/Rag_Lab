# core/chunking.py

from typing import List, Callable, Dict


def fixed_size_chunker(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Old behavior: fixed-size chunks with overlap.
    """
    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += max(1, chunk_size - chunk_overlap)

    return chunks


def paragraph_chunker(text: str, max_chars: int = 1200) -> List[str]:
    """
    Simple paragraph-based chunking.
    Splits on blank-lines / double newlines and then merges paragraphs
    until max_chars is reached.
    """
    raw_parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []

    current = ""
    for p in raw_parts:
        if not current:
            current = p
        elif len(current) + 2 + len(p) <= max_chars:
            current = current + "\n\n" + p
        else:
            chunks.append(current)
            current = p

    if current:
        chunks.append(current)

    return chunks


# Registry: key → chunking function
CHUNKERS: Dict[str, Callable[..., List[str]]] = {
    "fixed": fixed_size_chunker,
    "paragraph": paragraph_chunker,
}


def get_chunker(name: str) -> Callable[..., List[str]]:
    if name not in CHUNKERS:
        raise ValueError("Unknown chunker '%s'. Available: %s" % (name, list(CHUNKERS.keys())))
    return CHUNKERS[name]
