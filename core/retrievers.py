# core/retrievers.py

from typing import Dict, List

from core.embeddings import embed_texts
from core.vectordbs import get_vector_store


def retrieve(
    query: str,
    collection_name: str = "demo",
    k: int = 4,
    db_type: str = "chroma",
    embedding_backend: str = None,
) -> List[Dict]:
    """
    Embed the query and perform a similarity search in the chosen vector DB.
    Returns a list of dicts with document, metadata, and distance.
    """
    store = get_vector_store(db_type=db_type, collection_name=collection_name)
    query_embedding = embed_texts([query], backend_name=embedding_backend)[0]

    results = store.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs: List[Dict] = []
    for idx in range(len(results["ids"][0])):
        docs.append(
            {
                "id": results["ids"][0][idx],
                "document": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "distance": results["distances"][0][idx],
            }
        )

    return docs
