# core/vectordbs.py

import chromadb
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from config.settings import settings

# Global Chroma client (persistent)
_chroma_client = None  # type: chromadb.PersistentClient


class ChromaVectorStore(object):
    """
    Thin wrapper around a Chroma collection to present a uniform interface.
    """

    def __init__(self, collection_name: str):
        global _chroma_client
        if _chroma_client is None:
            settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))

        self.collection = _chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids,
        documents,
        metadatas,
        embeddings,
    ):
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(self, query_embeddings, n_results=4, include=None):
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=include or ["documents", "metadatas", "distances"],
        )


class QdrantVectorStore(object):
    """
    Wrapper around Qdrant so it behaves like our Chroma wrapper.
    We store:
      - vector: embedding
      - payload: {"document": text, ...metadata}
    """

    def __init__(self, collection_name: str):
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        # Optional prefix to avoid name collisions
        self.collection_name = settings.qdrant_collection_prefix + collection_name

    def _ensure_collection(self, vector_size: int):
        """
        Create collection lazily on first add if it doesn't exist.
        """
        try:
            self.client.get_collection(self.collection_name)
            # exists
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def add(
        self,
        ids,
        documents,
        metadatas,
        embeddings,
    ):
        if not embeddings:
            return

        vector_size = len(embeddings[0])
        self._ensure_collection(vector_size)

        points = []
        for idx, emb in enumerate(embeddings):
            meta = metadatas[idx] if metadatas and idx < len(metadatas) else {}
            doc = documents[idx] if documents and idx < len(documents) else ""
            payload = {"document": doc}
            if isinstance(meta, dict):
                payload.update(meta)

            points.append(
                PointStruct(
                    id=ids[idx],
                    vector=emb,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def query(self, query_embeddings, n_results=4, include=None):
        """
        Return a dict with keys: ids, documents, metadatas, distances
        shaped similarly to Chroma's query() output.
        """
        if not query_embeddings:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_vector = query_embeddings[0]

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=n_results,
            with_payload=True,
            with_vectors=False,
        )

        ids = []
        documents = []
        metadatas = []
        distances = []

        for hit in hits:
            ids.append(str(hit.id))
            payload = hit.payload or {}
            documents.append(payload.get("document", ""))
            # remove document from metadata
            meta = dict(payload)
            meta.pop("document", None)
            metadatas.append(meta)
            # Qdrant returns score; with cosine config this is similarity, not distance.
            # We invert it approximately so "smaller is better" like a distance.
            distances.append(1.0 - float(hit.score) if hit.score is not None else 0.0)

        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }


# Registry of vector DBs
AVAILABLE_VECTOR_DBS = ["chroma", "qdrant"]


def get_vector_store(db_type: str, collection_name: str):
    """
    Factory to get a vector store by type.
    """
    db_type = db_type.lower()
    if db_type == "chroma":
        return ChromaVectorStore(collection_name)
    if db_type == "qdrant":
        return QdrantVectorStore(collection_name)
    raise ValueError("Unsupported vector DB '%s'. Supported: %s" % (db_type, AVAILABLE_VECTOR_DBS))
