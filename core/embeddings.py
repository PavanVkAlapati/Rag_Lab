# core/embeddings.py

from typing import List, Dict
from openai import OpenAI

from config.settings import settings

_client = None  # type: OpenAI

# Registry of embedding backends you can expose in the UI
# Key = backend name shown to user
AVAILABLE_EMBEDDINGS: Dict[str, str] = {
    "openai_small": "text-embedding-3-small",
    "openai_large": "text-embedding-3-large",
    # You can add more later, e.g. custom models
    # "openai_legacy": "text-embedding-ada-002",
}


def get_embedding_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def get_embedding_model_name(backend_name: str = None) -> str:
    """
    Resolve a backend name (like 'openai_small') to a real OpenAI model name.
    If backend_name is None, fall back to settings.embedding_model.
    """
    if backend_name:
        model = AVAILABLE_EMBEDDINGS.get(backend_name)
        if model is None:
            raise ValueError(
                "Unknown embedding backend '%s'. "
                "Known backends: %s" % (backend_name, list(AVAILABLE_EMBEDDINGS.keys()))
            )
        return model

    # Fallback: whatever is in settings
    return settings.embedding_model


def embed_texts(texts: List[str], backend_name: str = None) -> List[List[float]]:
    """
    Returns a list of embeddings (one per input text).
    backend_name: key from AVAILABLE_EMBEDDINGS, or None to use settings.embedding_model.
    """
    if not texts:
        return []

    model_name = get_embedding_model_name(backend_name)
    client = get_embedding_client()
    response = client.embeddings.create(
        model=model_name,
        input=texts,
    )
    return [item.embedding for item in response.data]
