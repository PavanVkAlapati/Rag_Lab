# core/ingest.py

from pathlib import Path
from typing import List

from config.settings import settings
from core.embeddings import embed_texts
from core.vectordbs import get_vector_store
from core.chunking import get_chunker


def ingest_file(
    filename: str,
    collection_name: str = "demo",
    db_type: str = "chroma",
    chunker_name: str = "fixed",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_backend: str = None,
) -> None:
    """
    Ingest a single text file into the chosen vector DB.

    NOTE:
    - chunker_name: controls how text is split (see core.chunking)
    - embedding_backend: which embedding backend to use (core.embeddings)
    - db_type: which vector DB (currently only 'chroma')
    """
    data_path = Path(settings.data_dir) / filename
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    text = data_path.read_text(encoding="utf-8")

    chunker = get_chunker(chunker_name)
    if chunker_name == "fixed":
        chunks: List[str] = chunker(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        # For 'paragraph' and any others that don't use size/overlap
        chunks = chunker(text)

    if not chunks:
        print(f"No text chunks created from file {filename}")
        return

    embeddings = embed_texts(chunks, backend_name=embedding_backend)
    store = get_vector_store(db_type=db_type, collection_name=collection_name)

    ids = [f"{filename}-{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": filename,
            "chunk_id": i,
            "chunker": chunker_name,
            "embedding_backend": embedding_backend or "settings_default",
            "db_type": db_type,
        }
        for i in range(len(chunks))
    ]

    store.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(
        f"Ingested {len(chunks)} chunks from {filename} "
        f"into collection '{collection_name}' using db='{db_type}', "
        f"chunker='{chunker_name}', embedding='{embedding_backend or 'settings_default'}'."
    )


if __name__ == "__main__":
    # Default demo ingest; you can tweak parameters here
    ingest_file(
        "demo.txt",
        collection_name="demo",
        db_type="chroma",
        chunker_name="fixed",   # or "paragraph"
        chunk_size=1000,
        chunk_overlap=200,
        embedding_backend="openai_small",  # must exist in AVAILABLE_EMBEDDINGS or None
    )
