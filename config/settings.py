import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------
# Load .env from the project root (search_rag/.env)
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"

# Load environment variables
load_dotenv(dotenv_path=ENV_PATH)


class Settings:
    """
    Lightweight settings loader without Pydantic.
    Reads environment variables manually and sets defaults.
    """

    def __init__(self):
        # ---------------------------
        # Required: OpenAI API key
        # ---------------------------
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise RuntimeError(
                f"OPENAI_API_KEY is missing.\n"
                f"Expected in .env file at: {ENV_PATH}"
            )

        # ---------------------------
        # Optional: LLM + embedding model
        # ---------------------------
        self.llm_model_openai = os.getenv("LLM_MODEL_OPENAI", "gpt-4o-mini")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        # ---------------------------
        # Directory paths
        # ---------------------------
        self.base_dir: Path = BASE_DIR
        self.data_dir: Path = BASE_DIR / "data"
        self.chroma_persist_dir: Path = BASE_DIR / "store" / "chroma"

        # Ensure needed folders exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

        # ---------------------------
        # Qdrant configuration
        # Default local instance: http://localhost:6333
        # ---------------------------
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        self.qdrant_collection_prefix = os.getenv(
            "QDRANT_COLLECTION_PREFIX",
            "vanilla_rag_",  # prefix ensures isolation per project
        )


@lru_cache
def get_settings():
    """Singleton settings object."""
    return Settings()


settings = get_settings()
