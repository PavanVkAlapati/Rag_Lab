import os
import base64
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from io import BytesIO
import html
import re

import requests
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from openai import OpenAI

import pandas as pd
import numpy as np

# Optional PDF export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ============================================================
#                  PATHS & ASSETS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

LOGO_PATH = "assets/logo.png"
FAVICON_PATH = "assets/favicon.png"
SETTINGS_ICON_PATH = "assets/settings.png"
UPLOAD_ICON_PATH = "assets/upload.png"
FILL_ICON_PATH = "assets/fill.png"
INSPECT_ICON_PATH = "assets/inspect.png"
DOWNLOAD_ICON_PATH = "assets/download.png"
BOT_ICON_PATH = "assets/bot.png"
USER_ICON_PATH = "assets/user.png"


def load_image_b64(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


BOT_ICON_B64 = load_image_b64(BOT_ICON_PATH)
USER_ICON_B64 = load_image_b64(USER_ICON_PATH)

# ============================================================
#               LOAD .env AND CONFIGURE OPENAI
# ============================================================

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai_client = OpenAI(api_key=OPENAI_KEY)
else:
    openai_client = None

OPENAI_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_SEED = int(os.getenv("LLM_SEED", "42"))

# ============================================================
#                CONFIG & GLOBAL CONSTANTS
# ============================================================

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_BASE = os.getenv("QDRANT_COLLECTION", "rag_chat_collection")

MAX_HISTORY_TURNS = 12

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / "interaction_logs.xlsx"

CHROMA_DIR = BASE_DIR / "chroma_db"
CHROMA_COLLECTION_BASE = "rag_chat_collection"

FAISS_INDEX = None
FAISS_DIM = None
FAISS_PAYLOADS: List[Dict[str, Any]] = []

BGE_ENCODER = None
BGE_RERANKER = None

# URL regex
URL_REGEX = re.compile(r"https?://[^\s)\"'>]+")

# ============================================================
#            STREAMLIT PAGE CONFIG & CUSTOM THEME
# ============================================================

st.set_page_config(
    page_title="RAG Lab – Pluggable Chat",
    page_icon=str(FAVICON_PATH) if FAVICON_PATH else "🧪",
    layout="wide",
)

APP_CSS = f"""
<style>
/* ----- Global layout ----- */
main[data-testid="stAppViewContainer"] {{
    background: radial-gradient(circle at top left, #020617 0, #020617 35%, #020617 100%);
}}

section.main > div {{
    padding-top: 0.75rem;
}}

.block-container {{
    max-width: 1160px;
    padding-top: 0.9rem;
}}

/* ----- Sidebar ----- */
[data-testid="stSidebar"] {{
    background: radial-gradient(circle at top left, #020617 0%, #020617 55%, #020617 100%);
    color: #e5f4ff;
    border-right: 1px solid rgba(148,163,184,0.4);
}}

[data-testid="stSidebar"] .sidebar-content {{
    padding-top: 0.5rem;
}}

.rag-sidebar-title {{
    font-weight: 700;
    letter-spacing: 0.12em;
    font-size: 0.9rem;
}}

.rag-sidebar-sub {{
    font-size: 0.74rem;
    color: #9ca3af;
}}

.rag-sidebar-section-title {{
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.7rem;
    margin-bottom: 0.15rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
    border-top: 1px solid rgba(55,65,81,0.7);
    padding-top: 0.55rem;
}}

/* Streamlit widgets in sidebar */
[data-testid="stSidebar"] .stSelectbox > div > div {{
    border-radius: 999px;
    background: radial-gradient(circle at 0% 0%, rgba(15,23,42,0.96), rgba(15,23,42,0.9));
    border: 1px solid rgba(148,163,184,0.7);
    color: #e5e7eb;
}}
[data-testid="stSidebar"] .stSlider > div > div > div {{
    color: #e5e7eb;
}}
[data-testid="stSidebar"] .stSlider [role="slider"] {{
    box-shadow: 0 0 0 3px rgba(56,189,248,0.7);
}}
[data-testid="stSidebar"] .stDownloadButton button {{
    border-radius: 999px !important;
}}

/* Export button */
.rag-export-btn button {{
    width: 100%;
    border-radius: 999px !important;
    background: radial-gradient(circle at 10% 0%, rgba(34,211,238,0.35), rgba(15,23,42,1));
    border: 1px solid rgba(45,212,191,0.9);
    color: #ecfeff;
    font-weight: 600;
    box-shadow: 0 16px 35px rgba(15,23,42,0.85);
}}
.rag-export-btn button:disabled {{
    opacity: 0.35;
    box-shadow: none;
}}

/* ----- Liquid-glass top header ----- */
.rag-header {{ 
    text-align: center;
    margin-top: 2.0rem;
    margin-bottom: 1.0rem;
}}
.rag-header-inner {{
    display: inline-flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.85rem 1.5rem;
    border-radius: 999px;
    background: radial-gradient(circle at 10% 0%, rgba(34,211,238,0.28), rgba(15,23,42,0.98));
    box-shadow: 0 20px 55px rgba(15,23,42,0.95);
    backdrop-filter: blur(22px);
    border: 1px solid rgba(148,163,184,0.65);
}}
.rag-header-title {{
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #f9fafb;
    text-shadow: 0 0 14px rgba(56,189,248,0.85);
}}
.rag-header-sub {{
    font-size: 0.8rem;
    color: #dbeafe;
}}

/* Info banner */
.rag-banner {{
    margin-bottom: 0.6rem;
    padding: 0.75rem 1rem;
    border-radius: 18px;
    background: radial-gradient(circle at top left, rgba(15,23,42,0.98), rgba(15,23,42,0.96));
    color: #e5f4ff;
    border: 1px solid rgba(148,163,184,0.6);
    box-shadow: 0 18px 40px rgba(15,23,42,0.95);
    font-size: 0.85rem;
}}
.rag-banner strong {{
    color: #7dd3fc;
}}

/* ----- Section cards ----- */
.rag-section {{
    margin-top: 0.0rem;
    margin-bottom: 1.0rem;
    padding: 0.85rem 1.25rem 1.1rem;
    border-radius: 22px;
    background: radial-gradient(circle at 10% 10%, rgba(15,23,42,0.98), rgba(15,23,42,0.94));
    border: 1px solid rgba(30,64,175,0.8);
    box-shadow: 0 18px 40px rgba(15,23,42,0.9);
    backdrop-filter: blur(18px);
}}
.rag-section h2 {{
    font-size: 1.08rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
    color: #f9fafb;
}}
.rag-section p.help {{
    margin-top: 0;
    color: #cbd5f5;
    font-size: 0.86rem;
}}

/* ----- Upload panel ----- */
.rag-upload-row {{
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin-bottom: 0.25rem;
}}
.rag-upload-text-title {{
    font-weight: 600;
    color: #f9fafb;
}}
.rag-upload-text-sub {{
    font-size: 0.8rem;
    color: #cbd5f5;
}}

/* File uploader glass styling — DARK + readable */
[data-testid="stFileUploader"] > section {{
    border-radius: 18px;
    border: 1px dashed rgba(56,189,248,0.7);
    background: radial-gradient(circle at 0% 0%, rgba(15,23,42,0.95), rgba(15,23,42,0.90));
    box-shadow: 0 12px 30px rgba(15,23,42,0.6);
}}

/* Force ALL text inside uploader to be readable */
[data-testid="stFileUploader"] * {{
    color: #e5f4ff !important;
}}

/* Make the “Browse files” button readable too */
[data-testid="stFileUploader"] button {{
    color: #e5f4ff !important;
}}

/* Primary buttons on main page */
.stButton > button {{
    border-radius: 999px;
    padding: 0.45rem 1.3rem;
    font-weight: 600;
    background: radial-gradient(circle at 10% 0%, rgba(34,211,238,0.8), rgba(8,47,73,0.95));
    border: 1px solid rgba(56,189,248,0.9);
    color: #ecfeff;
    box-shadow: 0 14px 34px rgba(15,23,42,0.85);
}}
.stButton > button:disabled {{
    opacity: 0.45;
    box-shadow: none;
}}

/* ----- Chat layout ----- */
.chat-container {{
    margin-top: 0.35rem;
}}
.chat-row {{
    display: flex;
    margin-bottom: 0.55rem;
}}
.chat-row-bot {{
    justify-content: flex-start;
}}
.chat-row-user {{
    justify-content: flex-end;
}}
.chat-bubble {{
    max-width: 76%;
    display: flex;
    gap: 0.6rem;
}}
.chat-bubble-user {{
    flex-direction: row-reverse;
}}
.chat-avatar img {{
    width: 34px;
    height: 34px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(15,23,42,0.8);
    object-fit: cover;
}}
.chat-bubble-inner {{
    padding: 0.7rem 0.95rem;
    border-radius: 18px;
    font-size: 0.9rem;
    line-height: 1.45;
    word-wrap: break-word;
}}

/* Bot bubble (left) */
.chat-bubble-bot .chat-bubble-inner {{
    background: radial-gradient(circle at 0% 0%, rgba(15,23,42,0.97), rgba(15,23,42,0.94));
    color: #e5f4ff;
    border: 1px solid rgba(148,163,184,0.55);
    box-shadow: 0 18px 40px rgba(15,23,42,1);
}}

/* User bubble (right) */
.chat-bubble-user .chat-bubble-inner {{
    background: radial-gradient(circle at 10% 0%, rgba(34,211,238,0.24), rgba(15,23,42,0.94));
    color: #e0f9ff;
    border: 1px solid rgba(56,189,248,0.8);
    box-shadow: 0 18px 40px rgba(14,165,233,0.9);
}}

/* Chat input styling */
div[data-baseweb="textarea"] textarea {{
    border-radius: 999px !important;
}}
[data-testid="stChatInput"] > div {{
    background: radial-gradient(circle at 10% 0%, rgba(34,211,238,0.13), rgba(15,23,42,0.96));
    border-radius: 999px !important;
    border: 1px solid rgba(56,189,248,0.55);
    box-shadow: 0 16px 36px rgba(15,23,42,0.9);
}}

/* Inspect expander */
.rag-inspect-expander > details {{
    border-radius: 14px !important;
    background: linear-gradient(135deg, rgba(15,23,42,0.97), rgba(8,47,73,0.97));
    border: 1px solid rgba(56,189,248,0.55);
    color: #e0f2fe;
}}
.rag-inspect-expander summary {{
    font-size: 0.85rem;
    font-weight: 500;
}}

/* Images under assistant messages */
.rag-ref-images img {{
    border-radius: 14px;
    box-shadow: 0 16px 36px rgba(15,23,42,0.9);
    border: 1px solid rgba(148,163,184,0.7);
}}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)

# ============================================================
#                      QDRANT HELPERS
# ============================================================

def get_qdrant_client() -> QdrantClient:
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)

def get_qdrant_collection_name(encoder_choice: str) -> str:
    suffix = encoder_choice.lower()
    return f"{QDRANT_COLLECTION_BASE}_{suffix}"

# ----------------------- PATCH 1 ----------------------------
def ensure_collection(client: QdrantClient, vector_size: int, collection_name: str) -> None:
    """
    Ensure collection exists and matches vector_size.
    - If missing: create.
    - If exists with different dim: raise (do NOT recreate silently).
    """
    try:
        info = client.get_collection(collection_name)
        existing_size = info.config.params.vectors.size
        if existing_size != vector_size:
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' exists with dim={existing_size}, "
                f"but current embedding dim={vector_size}. "
                f"Pick matching encoder_choice or drop/recreate this collection manually."
            )
        return
    except RuntimeError:
        # mismatch -> do NOT recreate
        raise
    except Exception:
        # missing -> create
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
        )

# ============================================================
#                    EMBEDDINGS
# ============================================================

def get_bge_encoder():
    global BGE_ENCODER
    if BGE_ENCODER is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError("Install sentence-transformers for BGE: pip install sentence-transformers")
        BGE_ENCODER = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return BGE_ENCODER

def embed_openai(text: str, model: str) -> List[float]:
    if openai_client is None:
        raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY in your environment/.env.")
    text = text.replace("\n", " ")
    max_chars_for_embedding = 24000
    if len(text) > max_chars_for_embedding:
        text = text[:max_chars_for_embedding]

    resp = openai_client.embeddings.create(model=model, input=[text])
    return resp.data[0].embedding

def embed_bge(text: str) -> List[float]:
    model = get_bge_encoder()
    emb = model.encode(text, normalize_embeddings=True)
    return emb.tolist()

# ----------------------- PATCH 2 ----------------------------
def get_embedding(text: str, encoder_choice: str) -> List[float]:
    """
    Hybrid must be stable-dimension.
    Require BOTH OpenAI + BGE to succeed, else raise.
    """
    encoder_choice = encoder_choice.lower()

    if encoder_choice == "openai":
        return embed_openai(text, OPENAI_EMBED_MODEL)

    if encoder_choice == "bge":
        return embed_bge(text)

    if encoder_choice == "hybrid":
        # both MUST succeed to keep one fixed dimension
        openai_emb = embed_openai(text, OPENAI_EMBED_MODEL)
        bge_emb = embed_bge(text)
        return openai_emb + bge_emb

    return embed_openai(text, OPENAI_EMBED_MODEL)

# ============================================================
#                         CHUNKING
# ============================================================

def chunk_text_fixed(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def chunk_text_recursive(text: str, base_size: int = 1000, overlap: int = 200) -> List[str]:
    if len(text) <= base_size:
        return [text]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return chunk_text_fixed(text, chunk_size=base_size, overlap=overlap)

    chunks = []
    current = ""
    for p in paragraphs:
        candidate = (current + "\n\n" + p) if current else p
        if len(candidate) <= base_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)

    final_chunks = []
    for ch in chunks:
        if len(ch) > base_size:
            final_chunks.extend(chunk_text_fixed(ch, chunk_size=base_size, overlap=overlap))
        else:
            final_chunks.append(ch)
    return final_chunks

def chunk_text_semantic(text: str, max_chars: int = 1200) -> List[str]:
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    chunks = []
    current = ""
    for sent in sentences:
        if not sent.endswith("."):
            sent = sent + "."
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks

def chunk_text_paragraph(text: str, max_chars: int = 1000, overlap_paragraphs: int = 1) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []

    for p in paragraphs:
        if len(p) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
            for i in range(0, len(p), max_chars):
                segment = p[i : i + max_chars]
                chunks.append(segment)
            continue

        candidate = ("\n\n".join(current + [p])).strip()
        if len(candidate) <= max_chars:
            current.append(p)
        else:
            if current:
                chunks.append("\n\n".join(current))
            current = [p]

    if current:
        chunks.append("\n\n".join(current))

    if overlap_paragraphs > 0 and len(chunks) > 1:
        overlapped: List[str] = []
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped.append(ch)
            else:
                prev_paras = chunks[i - 1].split("\n\n")
                overlap = "\n\n".join(prev_paras[-overlap_paragraphs:])
                overlapped.append((overlap + "\n\n" + ch).strip())
        return overlapped

    return chunks

def apply_chunker(text: str, mode: str) -> List[str]:
    mode = mode.lower()
    if mode == "fixed":
        return chunk_text_fixed(text)
    if mode == "recursive":
        return chunk_text_recursive(text)
    if mode == "semantic":
        return chunk_text_semantic(text)
    return chunk_text_paragraph(text)

# ============================================================
#                    FILE HANDLING
# ============================================================

def load_file_to_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    uploaded_file.seek(0)

    if suffix in [".txt", ".md"]:
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            return "ERROR: pdfplumber not installed. Run: pip install pdfplumber"
        text = []
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)

    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError:
            return "ERROR: python-docx not installed. Run: pip install python-docx"
        doc = Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)

    if suffix == ".csv":
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        return df.to_csv(index=False)

    if suffix == ".json":
        try:
            data = json.load(uploaded_file)
        except Exception:
            return uploaded_file.read().decode("utf-8", errors="ignore")
        return json.dumps(data, indent=2)

    return uploaded_file.read().decode("utf-8", errors="ignore")

def is_image_file(uploaded_file) -> bool:
    suffix = Path(uploaded_file.name).suffix.lower()
    return suffix in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"]

def encode_image_to_b64(uploaded_file) -> str:
    uploaded_file.seek(0)
    img_bytes = uploaded_file.read()
    return base64.b64encode(img_bytes).decode("utf-8")

# ----------------- URL EXPANSION HELPERS --------------------

def extract_urls(text: str) -> List[str]:
    if not text:
        return []
    return URL_REGEX.findall(text)

def fetch_url_text(url: str, max_chars: int = 25000) -> str:
    try:
        resp = requests.get(url, timeout=10)
    except Exception:
        return ""

    if resp.status_code != 200:
        return ""

    content_type = (resp.headers.get("content-type") or "").lower()

    try:
        content_len = int(resp.headers.get("content-length", "0"))
    except ValueError:
        content_len = 0
    max_bytes = 5 * 1024 * 1024  # 5 MB
    if content_len and content_len > max_bytes:
        return ""

    if "text/html" in content_type:
        html_content = resp.text
        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(html_content, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
        except Exception:
            stripped = re.sub(r"<[^>]+>", " ", html_content)
            text = re.sub(r"\s+", " ", stripped)
        return text[:max_chars]

    if "pdf" in content_type:
        try:
            import pdfplumber
        except ImportError:
            return ""
        try:
            pdf_file = BytesIO(resp.content)
            text_pages: List[str] = []
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    text_pages.append(page.extract_text() or "")
            full_text = "\n".join(text_pages)
            return full_text[:max_chars]
        except Exception:
            return ""

    try:
        text = resp.text
    except Exception:
        return ""
    return text[:max_chars]

def expand_texts_with_links(
    texts: List[str],
    text_sources: List[str],
    follow_links: bool,
    max_urls_per_doc: int = 5,
) -> Tuple[List[str], List[str]]:
    if not follow_links:
        return texts, text_sources

    expanded_texts: List[str] = list(texts)
    expanded_sources: List[str] = list(text_sources)

    for idx, (doc_text, src) in enumerate(zip(texts, text_sources)):
        urls = extract_urls(doc_text)
        if not urls:
            continue
        count = 0
        for url in urls:
            if count >= max_urls_per_doc:
                break
            url_text = fetch_url_text(url)
            if not url_text.strip():
                continue
            expanded_texts.append(url_text)
            expanded_sources.append(f"{src} :: {url}")
            count += 1

    return expanded_texts, expanded_sources

# ============================================================
#              INGEST INTO VDBs (QDRANT / CHROMA / FAISS)
# ============================================================

# ----------------------- PATCH 3 ----------------------------
def ingest_documents_qdrant(
    client: QdrantClient,
    texts: List[str],
    text_sources: List[str],
    image_payloads: List[Dict[str, Any]],
    chunker_mode: str,
    encoder_choice: str,
) -> None:
    """
    Ensure collection ONCE per ingest; lock vector size from first embedding.
    Prevents Qdrant corruption from mixed dims.
    """
    points: List[qm.PointStruct] = []
    collection_name = get_qdrant_collection_name(encoder_choice)

    vector_size_locked = None

    # text docs
    for i, doc_text in enumerate(texts):
        if not doc_text or doc_text.startswith("ERROR:"):
            continue

        src = text_sources[i] if text_sources and i < len(text_sources) else f"doc_{i}"
        chunks = apply_chunker(doc_text, chunker_mode)

        for ch in chunks:
            if not ch.strip():
                continue

            embedding = get_embedding(ch, encoder_choice)

            if vector_size_locked is None:
                vector_size_locked = len(embedding)
                ensure_collection(client, vector_size_locked, collection_name)
            elif len(embedding) != vector_size_locked:
                raise RuntimeError(
                    f"Embedding dim mismatch in ingest. Expected {vector_size_locked}, got {len(embedding)}. "
                    f"encoder_choice='{encoder_choice}'."
                )

            pid = str(uuid.uuid4())
            payload = {"type": "text", "text": ch, "source": src}
            points.append(qm.PointStruct(id=pid, vector=embedding, payload=payload))

    # images
    for k, img in enumerate(image_payloads):
        caption = img.get("caption") or f"Image file: {img.get('source', f'image_{k}')}"
        embedding = get_embedding(caption, encoder_choice)

        if vector_size_locked is None:
            vector_size_locked = len(embedding)
            ensure_collection(client, vector_size_locked, collection_name)
        elif len(embedding) != vector_size_locked:
            raise RuntimeError(
                f"Embedding dim mismatch in image ingest. Expected {vector_size_locked}, got {len(embedding)}. "
                f"encoder_choice='{encoder_choice}'."
            )

        pid = str(uuid.uuid4())
        payload = {
            "type": "image",
            "caption": caption,
            "image_b64": img.get("image_b64"),
            "source": img.get("source"),
        }
        points.append(qm.PointStruct(id=pid, vector=embedding, payload=payload))

    if points:
        client.upsert(collection_name=collection_name, points=points)

def get_chroma_collection(encoder_choice: str):
    try:
        import chromadb
    except ImportError:
        raise RuntimeError("Install chromadb: pip install chromadb")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    coll_name = f"{CHROMA_COLLECTION_BASE}_{encoder_choice.lower()}"
    coll = client.get_or_create_collection(coll_name, embedding_function=None)
    return coll

def ingest_documents_chroma(
    texts: List[str],
    text_sources: List[str],
    image_payloads: List[Dict[str, Any]],
    chunker_mode: str,
    encoder_choice: str,
) -> None:
    coll = get_chroma_collection(encoder_choice)

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, doc_text in enumerate(texts):
        if not doc_text or doc_text.startswith("ERROR:"):
            continue
        src = (
            text_sources[i]
            if text_sources and i < len(text_sources)
            else f"doc_{i}"
        )
        chunks = apply_chunker(doc_text, chunker_mode)
        for ch in chunks:
            if not ch.strip():
                continue
            emb = get_embedding(ch, encoder_choice)
            pid = str(uuid.uuid4())
            ids.append(pid)
            embeddings.append(emb)
            documents.append(ch)
            metadatas.append({"type": "text", "source": src})

    for k, img in enumerate(image_payloads):
        caption = img.get("caption") or f"Image file: {img.get('source', f'image_{k}')}"
        emb = get_embedding(caption, encoder_choice)
        pid = str(uuid.uuid4())
        ids.append(pid)
        embeddings.append(emb)
        documents.append(caption)
        metadatas.append(
            {
                "type": "image",
                "source": img.get("source"),
                "image_b64": img.get("image_b64"),
                "caption": caption,
            }
        )

    if ids:
        coll.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

def ingest_documents_faiss(
    texts: List[str],
    text_sources: List[str],
    image_payloads: List[Dict[str, Any]],
    chunker_mode: str,
    encoder_choice: str,
) -> None:
    global FAISS_INDEX, FAISS_DIM, FAISS_PAYLOADS
    try:
        import faiss
    except ImportError:
        raise RuntimeError("Install faiss-cpu: pip install faiss-cpu")

    vectors = []
    payloads = []

    for i, doc_text in enumerate(texts):
        if not doc_text or doc_text.startswith("ERROR:"):
            continue
        src = (
            text_sources[i]
            if text_sources and i < len(text_sources)
            else f"doc_{i}"
        )
        chunks = apply_chunker(doc_text, chunker_mode)
        for ch in chunks:
            if not ch.strip():
                continue
            emb = get_embedding(ch, encoder_choice)
            vectors.append(emb)
            payloads.append({"type": "text", "text": ch, "source": src})

    for k, img in enumerate(image_payloads):
        caption = img.get("caption") or f"Image file: {img.get('source', f'image_{k}')}"
        emb = get_embedding(caption, encoder_choice)
        vectors.append(emb)
        payloads.append(
            {
                "type": "image",
                "caption": caption,
                "image_b64": img.get("image_b64"),
                "source": img.get("source"),
            }
        )

    if not vectors:
        return

    X = np.array(vectors, dtype="float32")
    dim = X.shape[1]

    if FAISS_INDEX is None:
        FAISS_INDEX = faiss.IndexFlatIP(dim)
        FAISS_DIM = dim
    else:
        if dim != FAISS_DIM:
            raise RuntimeError(
                f"FAISS index dim mismatch. Existing={FAISS_DIM}, new={dim}. "
                f"Restart app or clear index."
            )

    FAISS_INDEX.add(X)
    FAISS_PAYLOADS.extend(payloads)

def ingest_documents(
    vdb_choice: str,
    texts: List[str],
    text_sources: List[str],
    image_payloads: List[Dict[str, Any]],
    chunker_mode: str,
    encoder_choice: str,
    follow_links: bool,
):
    texts_expanded, sources_expanded = expand_texts_with_links(
        texts, text_sources, follow_links=follow_links
    )

    num_chunks = 0
    for t in texts_expanded:
        if t and not str(t).startswith("ERROR:"):
            try:
                num_chunks += len(apply_chunker(t, chunker_mode))
            except Exception:
                pass

    if vdb_choice == "Qdrant":
        client = get_qdrant_client()
        ingest_documents_qdrant(
            client,
            texts_expanded,
            sources_expanded,
            image_payloads,
            chunker_mode,
            encoder_choice,
        )
    elif vdb_choice == "Chroma":
        ingest_documents_chroma(
            texts_expanded,
            sources_expanded,
            image_payloads,
            chunker_mode,
            encoder_choice,
        )
    elif vdb_choice == "FAISS":
        ingest_documents_faiss(
            texts_expanded,
            sources_expanded,
            image_payloads,
            chunker_mode,
            encoder_choice,
        )
    else:
        raise RuntimeError(f"Unknown VDB choice: {vdb_choice}")

    return num_chunks

# ============================================================
#                    SEARCH FUNCTIONS
# ============================================================

def search_qdrant_http(
    query_embedding: List[float], top_k: int, encoder_choice: str
) -> List[Dict[str, Any]]:
    collection_name = get_qdrant_collection_name(encoder_choice)
    url = QDRANT_URL.rstrip("/") + f"/collections/{collection_name}/points/search"

    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY

    body = {"vector": query_embedding, "limit": top_k, "with_payload": True}
    resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Qdrant HTTP search failed: {resp.status_code} {resp.text}")

    data = resp.json()
    result = data.get("result", [])
    docs: List[Dict[str, Any]] = []
    for r in result:
        payload = r.get("payload") or {}
        if payload:
            docs.append(payload)
    return docs

def search_qdrant(query: str, top_k: int, encoder_choice: str) -> List[Dict[str, Any]]:
    client = get_qdrant_client()
    query_embedding = get_embedding(query, encoder_choice)
    collection_name = get_qdrant_collection_name(encoder_choice)
    ensure_collection(client, len(query_embedding), collection_name)
    return search_qdrant_http(query_embedding, top_k=top_k, encoder_choice=encoder_choice)

def search_chroma(query: str, top_k: int, encoder_choice: str) -> List[Dict[str, Any]]:
    coll = get_chroma_collection(encoder_choice)
    q_emb = get_embedding(query, encoder_choice)
    res = coll.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    docs: List[Dict[str, Any]] = []
    if not res or "metadatas" not in res or not res["metadatas"]:
        return docs
    metas = res["metadatas"][0]
    docs_list = res.get("documents", [[]])[0]
    for md, doc_text in zip(metas, docs_list):
        payload = dict(md)
        if md.get("type") == "text":
            payload.setdefault("text", doc_text)
        docs.append(payload)
    return docs

def search_faiss(query: str, top_k: int, encoder_choice: str) -> List[Dict[str, Any]]:
    global FAISS_INDEX, FAISS_PAYLOADS, FAISS_DIM
    if FAISS_INDEX is None or not FAISS_PAYLOADS:
        return []

    q_emb = get_embedding(query, encoder_choice)
    if len(q_emb) != FAISS_DIM:
        raise RuntimeError(f"FAISS query dim {len(q_emb)} does not match index dim {FAISS_DIM}")

    X = np.array([q_emb], dtype="float32")
    k = min(top_k, len(FAISS_PAYLOADS))
    D, I = FAISS_INDEX.search(X, k)

    docs = []
    for idx in I[0]:
        docs.append(FAISS_PAYLOADS[idx])
    return docs

def search_vdb(
    vdb_choice: str,
    query: str,
    top_k: int,
    encoder_choice: str,
    hybrid_search: bool,
) -> List[Dict[str, Any]]:
    if vdb_choice == "Qdrant":
        docs = search_qdrant(query, top_k, encoder_choice)
    elif vdb_choice == "Chroma":
        docs = search_chroma(query, top_k, encoder_choice)
    elif vdb_choice == "FAISS":
        docs = search_faiss(query, top_k, encoder_choice)
    else:
        raise RuntimeError(f"Unknown VDB choice: {vdb_choice}")

    return docs

# ============================================================
#                        RERANKING
# ============================================================

def get_bge_reranker():
    global BGE_RERANKER
    if BGE_RERANKER is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise RuntimeError(
                "Install sentence-transformers for BGE reranker: pip install sentence-transformers"
            )
        BGE_RERANKER = CrossEncoder("BAAI/bge-reranker-base")
    return BGE_RERANKER

def rerank_cohere(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        import cohere
    except ImportError:
        raise RuntimeError("Install cohere: pip install cohere")

    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY not set in env/.env")

    co = cohere.Client(api_key)
    candidates = []
    for d in docs:
        txt = d.get("text") or d.get("caption") or ""
        candidates.append(txt)

    if not candidates:
        return docs

    res = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=candidates,
        top_n=len(candidates),
    )
    reordered = [docs[r.index] for r in res.results]
    return reordered

def rerank_bge(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    model = get_bge_reranker()
    pairs = []
    for d in docs:
        txt = d.get("text") or d.get("caption") or ""
        pairs.append((query, txt))
    if not pairs:
        return docs

    scores = model.predict(pairs)
    idxs = list(range(len(docs)))
    idxs_sorted = sorted(idxs, key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in idxs_sorted]

def simple_rerank(query: str, docs: List[Dict[str, Any]], reranker_mode: str) -> List[Dict[str, Any]]:
    reranker_mode = reranker_mode.lower()
    if not docs or reranker_mode == "none":
        return docs

    if reranker_mode.startswith("cohere"):
        return rerank_cohere(query, docs)

    if reranker_mode.startswith("bge"):
        return rerank_bge(query, docs)

    if reranker_mode.startswith("hybrid"):
        try:
            docs_co = rerank_cohere(query, docs)
        except Exception:
            docs_co = None
        try:
            docs_bge = rerank_bge(query, docs)
        except Exception:
            docs_bge = None

        if docs_co is None and docs_bge is None:
            return docs
        if docs_co is None:
            return docs_bge
        if docs_bge is None:
            return docs_co

        base = docs
        rank_co = {id(doc): i for i, doc in enumerate(docs_co)}
        rank_bg = {id(doc): i for i, doc in enumerate(docs_bge)}
        scores = []
        for doc in base:
            rc = rank_co.get(id(doc), len(base))
            rb = rank_bg.get(id(doc), len(base))
            scores.append((doc, rc + rb))
        scores_sorted = sorted(scores, key=lambda x: x[1])
        return [d for d, _ in scores_sorted]

    return docs

# ============================================================
#                     RAG CHAT PIPELINE
# ============================================================

def build_conversation_context(history: List[Dict[str, Any]], user_message: str) -> str:
    lines = []
    for msg in history:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
    lines.append(f"User: {user_message}")
    return "\n".join(lines)

def chat_with_rag(
    vdb_choice: str,
    user_message: str,
    history: List[Dict[str, Any]],
    top_k: int,
    encoder_choice: str,
    hybrid_search_flag: bool,
    reranker_mode: str,
) -> Tuple[str, List[str], List[Dict[str, Any]]]:

    docs = search_vdb(
        vdb_choice=vdb_choice,
        query=user_message,
        top_k=top_k,
        encoder_choice=encoder_choice,
        hybrid_search=hybrid_search_flag,
    )

    docs = simple_rerank(user_message, docs, reranker_mode)

    text_contexts: List[str] = []
    image_b64s: List[str] = []
    seen_texts = set()

    for d in docs:
        dtype = d.get("type")
        if dtype == "text":
            t = (d.get("text") or "").strip()
            if t and t not in seen_texts:
                text_contexts.append(t)
                seen_texts.add(t)
        elif dtype == "image":
            b64 = d.get("image_b64")
            if b64:
                image_b64s.append(b64)

    if not text_contexts and not image_b64s:
        return "I could not find this information in the document.", [], docs

    context_str = "\n\n".join(text_contexts[:5]).strip()

    if not context_str:
        return "I could not find this information in the document.", image_b64s, docs

    convo_str = build_conversation_context(history, user_message)

    system_prompt = (
        "You are a STRICT retrieval-augmented assistant.\n"
        "- ONLY answer using the provided context from the knowledge base.\n"
        "- If the context does NOT clearly contain the answer, reply exactly:\n"
        "  \"I could not find this information in the document.\"\n"
        "- Do NOT use outside knowledge.\n"
        "- Do NOT guess or assume.\n"
        "- If the user asks something not in context, say not found.\n"
        "- Never invent names, dates, numbers, or events."
    )

    if openai_client is None:
        answer = (
            "LLM is not configured (OPENAI_API_KEY missing). "
            "Here is the raw retrieved context:\n\n" + context_str
        )
        return answer, image_b64s, docs

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n{convo_str}\n\n"
                f"Retrieved context:\n{context_str}\n\n"
                "Answer ONLY from this context. "
                "If you cannot find a clear answer, respond exactly:\n"
                "\"I could not find this information in the document.\""
            ),
        },
    ]

    resp = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0,
        seed=LLM_SEED,
    )
    answer = (resp.choices[0].message.content or "").strip()

    if not answer:
        answer = "Upon verifying the source, I see the quesry asked is out of scope. Please reprompt or update the source"

    return answer, image_b64s, docs

# ============================================================
#                     METRICS & LOGGING
# ============================================================

def compute_mmr_score(docs: List[Dict[str, Any]], encoder_choice: str) -> float:
    texts = []
    for d in docs:
        txt = d.get("text") or d.get("caption") or ""
        txt = txt.strip()
        if txt:
            texts.append(txt[:2000])

    if len(texts) < 2:
        return 0.0

    embs = []
    for t in texts:
        try:
            embs.append(get_embedding(t, encoder_choice))
        except Exception:
            continue

    if len(embs) < 2:
        return 0.0

    E = np.array(embs, dtype="float32")
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    E_norm = E / norms
    sim_mat = E_norm @ E_norm.T

    n = sim_mat.shape[0]
    if n < 2:
        return 0.0

    mask = ~np.eye(n, dtype=bool)
    vals = sim_mat[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0

    avg_sim = float(vals.mean())
    mmr_score = 1.0 - avg_sim
    return mmr_score

def evaluate_answer_metrics_llm(
    question: str,
    answer: str,
    docs: List[Dict[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    metrics = {
        "precision_at_k": None,
        "hit_rate": None,
        "answer_relevance_score": None,
    }

    if openai_client is None or not docs:
        return metrics

    doc_snippets = []
    for i, d in enumerate(docs[:top_k]):
        txt = d.get("text") or d.get("caption") or ""
        txt = txt.strip()
        if not txt:
            continue
        doc_snippets.append(f"Doc {i+1}: {txt[:600]}")
    context_block = "\n\n".join(doc_snippets) or "(no retrieved docs)"

    prompt = f"""
You are evaluating a retrieval-augmented QA system.

Question:
{question}

Answer:
{answer}

Retrieved document snippets:
{context_block}

1. For each document, decide if it is relevant (1) or not relevant (0) to answering the question.
2. Then compute:
   - precision_at_k = (# relevant docs among retrieved docs) / max(1, number of retrieved docs).
   - hit_rate = 1 if at least one relevant doc, otherwise 0.
   - answer_relevance_score = a single score from 0 to 1 indicating how well the answer addresses the question given the retrieved context.

Respond ONLY with a JSON object like:
{{
  "precision_at_k": 0.8,
  "hit_rate": 1,
  "answer_relevance_score": 0.9
}}
"""
    try:
        resp = openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            seed=LLM_SEED,
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)

        metrics["precision_at_k"] = float(data.get("precision_at_k", 0.0))
        metrics["hit_rate"] = int(data.get("hit_rate", 0))
        metrics["answer_relevance_score"] = float(data.get("answer_relevance_score", 0.0))
    except Exception:
        pass

    return metrics

def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to RAG Lab – Liquid Glass Edition 🧪\n\n"
                    "1. Upload and index your data in section 1.\n"
                    "2. Ask questions in section 2.\n"
                    "3. Use **Inspect answer** to see retrieval quality and reference chunks.\n"
                    "4. Export chat from the sidebar."
                ),
                "images": [],
                "docs": [],
                "log_row": None,
            }
        ]

def reset_legacy_history_if_needed():
    hist = st.session_state.get("chat_history")
    if not hist:
        return

    if not isinstance(hist, list):
        st.session_state.pop("chat_history", None)
        init_session()
        return

    legacy = False
    for m in hist:
        if not isinstance(m, dict):
            txt = str(m)
            if "chat-bubble-inner" in txt or "<div" in txt:
                legacy = True
            else:
                legacy = True
            break

        if "role" not in m or "content" not in m:
            legacy = True
            break

        txt = str(m.get("content", ""))
        if "chat-bubble-inner" in txt or "<div" in txt:
            legacy = True
            break

    if legacy:
        st.session_state.pop("chat_history", None)
        init_session()

def log_interaction(
    question: str,
    answer: str,
    vdb_choice: str,
    embedding_choice: str,
    chunker_mode: str,
    reranker_mode: str,
    hybrid_search_flag: bool,
    docs: List[Dict[str, Any]],
    top_k: int,
    encoder_choice_internal: str,
) -> Dict[str, Any]:
    timestamp = datetime.utcnow().isoformat()
    session_id = st.session_state.get("session_id", "unknown")

    doc_sources = []
    for d in docs:
        src = d.get("source") or d.get("caption") or d.get("text", "")[:50]
        doc_sources.append(str(src))
    doc_sources_str = " | ".join(doc_sources)

    relevance_metrics = evaluate_answer_metrics_llm(question, answer, docs, top_k)
    precision_at_k = relevance_metrics.get("precision_at_k")
    hit_rate = relevance_metrics.get("hit_rate")
    answer_relevance_score = relevance_metrics.get("answer_relevance_score")
    recall_at_k = precision_at_k

    mmr_score = compute_mmr_score(docs, encoder_choice_internal)

    row = {
        "timestamp_utc": timestamp,
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "vdb": vdb_choice,
        "embedding_choice": embedding_choice,
        "chunker_mode": chunker_mode,
        "reranker_mode": reranker_mode,
        "hybrid_search": bool(hybrid_search_flag),
        "num_docs": len(docs),
        "doc_sources": doc_sources_str,
        "recall_at_k_approx": recall_at_k,
        "answer_relevance_score": answer_relevance_score,
        "mmr_score": mmr_score,
        "hit_rate": hit_rate,
    }

    try:
        if LOG_PATH.exists():
            df = pd.read_excel(LOG_PATH)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_excel(LOG_PATH, index=False)
    except Exception as e:
        print("Failed to write interaction log:", e)

    return row

# ============================================================
#                CHAT RENDERING & EXPORT HELPERS
# ============================================================

def escape_content_for_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")

def render_chat_history():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for idx, msg in enumerate(st.session_state.chat_history):
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            continue

        role = msg["role"]
        content = msg["content"]
        safe_content = escape_content_for_html(content)

        if role == "user":
            avatar_b64 = USER_ICON_B64
            row_class = "chat-row chat-row-user"
            bubble_class = "chat-bubble chat-bubble-user"
        else:
            avatar_b64 = BOT_ICON_B64
            row_class = "chat-row chat-row-bot"
            bubble_class = "chat-bubble chat-bubble-bot"

        avatar_html = ""
        if avatar_b64:
            avatar_html = (
                f'<div class="chat-avatar">'
                f'<img src="data:image/png;base64,{avatar_b64}" alt="avatar"></div>'
            )

        html_block = f"""
<div class="{row_class}">
  <div class="{bubble_class}">
    {avatar_html}
    <div class="chat-bubble-inner">{safe_content}</div>
  </div>
</div>
"""
        st.markdown(html_block, unsafe_allow_html=True)

        if role == "assistant":
            images_b64 = msg.get("images") or []
            if images_b64:
                cols = st.columns(len(images_b64))
                for i, (img_b64, col) in enumerate(zip(images_b64, cols)):
                    with col:
                        with st.container():
                            st.markdown('<div class="rag-ref-images">', unsafe_allow_html=True)
                            st.image(base64.b64decode(img_b64), caption=f"Reference {i+1}", width=165)
                            st.markdown("</div>", unsafe_allow_html=True)

            log_row = msg.get("log_row")
            docs = msg.get("docs") or []
            if log_row is not None:
                with st.expander("🔍 Inspect answer", expanded=False):
                    st.markdown('<div class="rag-inspect-expander">', unsafe_allow_html=True)
                    st.write("**Quality metrics**")
                    st.write(
                        f"- Approx. recall@k (from precision@k): "
                        f"{log_row.get('recall_at_k_approx') if log_row.get('recall_at_k_approx') is not None else 'N/A'}"
                    )
                    st.write(
                        f"- Answer relevance score (0–1): "
                        f"{log_row.get('answer_relevance_score') if log_row.get('answer_relevance_score') is not None else 'N/A'}"
                    )
                    st.write(
                        f"- MMR diversity score (0–1): "
                        f"{log_row.get('mmr_score') if log_row.get('mmr_score') is not None else 'N/A'}"
                    )
                    st.write(
                        f"- Hit rate (>=1 relevant doc): "
                        f"{log_row.get('hit_rate') if log_row.get('hit_rate') is not None else 'N/A'}"
                    )

                    st.markdown("---")
                    st.write("**Top-k retrieved references**")
                    if not docs:
                        st.write("_No retrieved documents recorded._")
                    else:
                        for i, d in enumerate(docs, start=1):
                            src = d.get("source") or "(unknown source)"
                            dtype = d.get("type") or "text"
                            snippet = d.get("text") or d.get("caption") or ""
                            snippet = snippet.strip().replace("\n", " ")
                            if len(snippet) > 260:
                                snippet = snippet[:260] + "…"
                            st.markdown(
                                f"**{i}. [{dtype}]** — `{src}`  \n"
                                f"> {snippet}"
                            )
                    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

def build_markdown_export() -> Tuple[str, str]:
    if "chat_history" not in st.session_state:
        return "", "chat.md"

    lines = ["# RAG Lab – Chat Transcript\n"]
    for msg in st.session_state.chat_history:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            continue
        role = msg["role"]
        header = "User" if role == "user" else "Assistant"
        lines.append(f"## {header}\n")
        lines.append(msg["content"])
        lines.append("")
    content = "\n".join(lines)
    fname = f"rag_lab_chat_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
    return content, fname

def build_pdf_export() -> Tuple[bytes, str]:
    if not REPORTLAB_AVAILABLE or "chat_history" not in st.session_state:
        return b"", "chat.pdf"

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x_margin = 50
    y = height - 50

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x_margin, y, "RAG Lab – Chat Transcript")
    y -= 30

    c.setFont("Helvetica", 10)
    for msg in st.session_state.chat_history:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            continue
        role = "User" if msg["role"] == "user" else "Assistant"
        text = f"{role}: {msg['content']}"
        wrapped = []
        for line in text.split("\n"):
            while len(line) > 95:
                wrapped.append(line[:95])
                line = line[95:]
            wrapped.append(line)
        for line in wrapped:
            if y < 60:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 50
            c.drawString(x_margin, y, line)
            y -= 14
        y -= 10

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    fname = f"rag_lab_chat_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    return pdf_bytes, fname

# ============================================================
#                          MAIN APP
# ============================================================

def main():
    init_session()
    reset_legacy_history_if_needed()

    with st.sidebar:
        cols_header = st.columns([0.25, 0.5, 0.25])
        with cols_header[0]:
            if LOGO_PATH:
                st.image(str(LOGO_PATH), width=40)
        with cols_header[1]:
            st.markdown(
                "<div class='rag-sidebar-title'>RAG LAB</div>"
                "<div class='rag-sidebar-sub'>Liquid Glass Playground</div>",
                unsafe_allow_html=True,
            )
        with cols_header[2]:
            if SETTINGS_ICON_PATH:
                st.image(str(SETTINGS_ICON_PATH), width=38)

        st.markdown("<div class='rag-sidebar-section-title'>Stack</div>", unsafe_allow_html=True)
        vdb_choice = st.selectbox("Vector DB", ["Qdrant", "Chroma", "FAISS"], index=0)
        embedding_choice = st.selectbox("Embedding encoder", ["OpenAI", "BGE", "Hybrid"], index=0)
        chunker_mode = st.selectbox("Chunker", ["Fixed", "Recursive", "Semantic", "Paragraph"], index=0)
        reranker_mode = st.selectbox("Reranker", ["None", "Cohere", "BGE", "Hybrid"], index=0)
        hybrid_search_flag = st.checkbox("Hybrid search flag (for logging only)", value=False)
        follow_links_flag = st.checkbox(
            "Follow links in docs during ingest",
            value=False,
            help="Fetch and index content behind URLs in uploaded text/PDFs.",
        )
        top_k = st.slider("Top-k retrieved", min_value=1, max_value=15, value=5)

        st.markdown("<div class='rag-sidebar-section-title'>Export Chat</div>", unsafe_allow_html=True)
        export_format = st.selectbox("Format", ["Markdown (.md)", "PDF (.pdf)"], index=0)

        has_chat = (
            "chat_history" in st.session_state
            and isinstance(st.session_state.chat_history, list)
            and len(st.session_state.chat_history) > 0
        )

        if export_format.startswith("Markdown"):
            content_md, fname_md = build_markdown_export()
            with st.container():
                st.markdown("<div class='rag-export-btn'>", unsafe_allow_html=True)
                st.download_button(
                    label="Download chat (.md)",
                    data=content_md,
                    file_name=fname_md,
                    mime="text/markdown",
                    disabled=not has_chat,
                    key="download_md",
                )
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            if REPORTLAB_AVAILABLE:
                pdf_bytes, fname_pdf = build_pdf_export()
            else:
                pdf_bytes, fname_pdf = b"", "chat.pdf"
            with st.container():
                st.markdown("<div class='rag-export-btn'>", unsafe_allow_html=True)
                st.download_button(
                    label="Download chat (.pdf)",
                    data=pdf_bytes,
                    file_name=fname_pdf,
                    mime="application/pdf",
                    disabled=(not has_chat or not REPORTLAB_AVAILABLE),
                    key="download_pdf",
                )
                st.markdown("</div>", unsafe_allow_html=True)
            if not REPORTLAB_AVAILABLE:
                st.caption("Install `reportlab` to enable PDF export.")

    header_logo_html = ""
    if LOGO_PATH:
        logo_b64 = load_image_b64(LOGO_PATH)
        if logo_b64:
            header_logo_html = (
                f"<img src='data:image/png;base64,{logo_b64}' "
                f"style='width:40px;height:40px;border-radius:14px;"
                f"box-shadow:0 12px 40px rgba(15,23,42,0.9);'/>"
            )

    st.markdown(
        f"""
<div class="rag-header">
  <div class="rag-header-inner">
    {header_logo_html}
    <div>
      <div class="rag-header-title">RAG Lab – Pluggable Chat</div>
      <div class="rag-header-sub">Upload → index → chat → inspect retrieval quality & export transcripts.</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if openai_client is None:
        st.error(
            "OPENAI_API_KEY not found – LLM and OpenAI embeddings will be unavailable. "
            "Set it in your `.env` file or environment for full functionality."
        )

    st.markdown(
        """
<div class="rag-banner">
  <strong>Flow:</strong> Configure your stack in the sidebar → Upload & index in <strong>1</strong> →
  Ask questions in <strong>2</strong> → Inspect retrieval → Export your transcript.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rag-section">', unsafe_allow_html=True)
    st.markdown("<h2>1. Upload and index your data</h2>", unsafe_allow_html=True)
    st.markdown(
        '<p class="help">Upload documents (text, PDFs, images, etc.) to build your knowledge base.</p>',
        unsafe_allow_html=True,
    )

    with st.container():
        cols_u = st.columns([0.14, 0.86])
        with cols_u[0]:
            if UPLOAD_ICON_PATH:
                st.image(str(UPLOAD_ICON_PATH), width=80)
        with cols_u[1]:
            st.markdown(
                """
<div class="rag-upload-row">
  <div>
    <div class="rag-upload-text-title">Drag & drop or browse your files</div>
    <div class="rag-upload-text-sub">
      Any text format, PDFs, JSON, CSV, DOCX, plus images for visual references.
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    uploaded_files = st.file_uploader(
        "Drop files here",
        type=None,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    index_button = st.button("Index to selected VDB", disabled=not uploaded_files)

    if index_button and uploaded_files:
        texts: List[str] = []
        text_sources: List[str] = []
        image_payloads: List[Dict[str, Any]] = []

        for f in uploaded_files:
            if is_image_file(f):
                image_payloads.append(
                    {
                        "source": f.name,
                        "image_b64": encode_image_to_b64(f),
                        "caption": None,
                    }
                )
            else:
                txt = load_file_to_text(f)
                texts.append(txt)
                text_sources.append(f.name)

        with st.spinner("Indexing into selected vector DB..."):
            if FILL_ICON_PATH:
                st.image(str(FILL_ICON_PATH), width=220)
            num_chunks = ingest_documents(
                vdb_choice=vdb_choice,
                texts=texts,
                text_sources=text_sources,
                image_payloads=image_payloads,
                chunker_mode=chunker_mode,
                encoder_choice=embedding_choice,
                follow_links=follow_links_flag,
            )
            st.session_state.index_ready = num_chunks > 0
            st.session_state.index_chunk_count = num_chunks
        st.success(f"Indexing completed. Indexed {st.session_state.get('index_chunk_count', 0)} chunks. You can now chat.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="rag-section">', unsafe_allow_html=True)
    st.markdown("<h2>2. Chat with your knowledge base</h2>", unsafe_allow_html=True)
    st.markdown(
        '<p class="help">Ask questions about your indexed documents. Answers come only from your data.</p>',
        unsafe_allow_html=True,
    )

    render_chat_history()

    user_input = st.chat_input("Ask something about your indexed documents…")

    if user_input:
        if not st.session_state.get("index_ready", False):
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input, "images": [], "docs": [], "log_row": None}
            )
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": "Nothing is indexed yet (or indexing failed). Upload documents and click Index first.",
                    "images": [],
                    "docs": [],
                    "log_row": None,
                }
            )
            st.rerun()

        st.session_state.chat_history.append(
            {"role": "user", "content": user_input, "images": [], "docs": [], "log_row": None}
        )

        history_for_llm = [
            m for m in st.session_state.chat_history
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
        ]
        if len(history_for_llm) > MAX_HISTORY_TURNS:
            history_for_llm = history_for_llm[-MAX_HISTORY_TURNS:]

        with st.spinner("Thinking with your knowledge base…"):
            answer, image_b64s, docs = chat_with_rag(
                vdb_choice=vdb_choice,
                user_message=user_input,
                history=history_for_llm[:-1],
                top_k=top_k,
                encoder_choice=embedding_choice,
                hybrid_search_flag=hybrid_search_flag,
                reranker_mode=reranker_mode,
            )

            log_row = log_interaction(
                question=user_input,
                answer=answer,
                vdb_choice=vdb_choice,
                embedding_choice=embedding_choice,
                chunker_mode=chunker_mode,
                reranker_mode=reranker_mode,
                hybrid_search_flag=hybrid_search_flag,
                docs=docs,
                top_k=top_k,
                encoder_choice_internal=embedding_choice.lower(),
            )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "images": image_b64s,
                "docs": docs,
                "log_row": log_row,
            }
        )

        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
