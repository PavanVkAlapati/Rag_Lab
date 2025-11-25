# RagLab

A Modular Platform for Retrieval-Augmented Generation Experimentation

## Overview

RagLab is a modular, extensible environment for developing, testing, and evaluating Retrieval-Augmented Generation (RAG) systems. It provides a full pipeline including ingestion, chunking, embedding, vector database storage, retrieval, optional reranking, context construction, and LLM answering.

The system includes a Streamlit-based UI, a FastAPI backend, multiple vector database options, switchable chunkers, multiple embedding backends, and an evaluation framework with Excel-based logging.

RagLab is designed for rapid iteration, reproducibility, and flexibility for research and production-level experimentation.

---

## Architecture Diagrams

The following diagrams describe the architecture and flow of RagLab.

### 1. Frontend to Backend Chat Flow

[https://www.figma.com/board/fimZiOJCSL4Suo3nqhpPHf/Frontend-to-Backend-Chat-Flow?node-id=0-1](https://www.figma.com/board/fimZiOJCSL4Suo3nqhpPHf/Frontend-to-Backend-Chat-Flow?node-id=0-1)

### 2. RAG Search and Tavily Summarization Flow

[https://www.figma.com/board/7yjXssB86LKjeXIaH5KnPt/RAG-Search-and-Tavily-Summarization-Flow?node-id=0-1](https://www.figma.com/board/7yjXssB86LKjeXIaH5KnPt/RAG-Search-and-Tavily-Summarization-Flow?node-id=0-1)

### 3. End-to-End System Architecture

[https://www.figma.com/board/hRWcmL4freuBkHfj53BnHh/End-to-End-System-Architecture?node-id=0-1](https://www.figma.com/board/hRWcmL4freuBkHfj53BnHh/End-to-End-System-Architecture?node-id=0-1)

### 4. Data Ingestion and Indexing Flow

[https://www.figma.com/board/1vvtP94q3yRTPmH7rL9wVK/Data-Ingestion-and-Indexing-Flow?node-id=0-1](https://www.figma.com/board/1vvtP94q3yRTPmH7rL9wVK/Data-Ingestion-and-Indexing-Flow?node-id=0-1)

### 5. Hybrid Retrieval and Summarization Flow

[https://www.figma.com/board/Xg2fMSWi7OiNA9NgwSX8C2/Hybrid-Retrieval-and-Summarization-Flow?node-id=0-1](https://www.figma.com/board/Xg2fMSWi7OiNA9NgwSX8C2/Hybrid-Retrieval-and-Summarization-Flow?node-id=0-1)

---

## Features

### Core Functionalities

* Flexible vector DB selection: ChromaDB or Qdrant
* Multiple chunking strategies: fixed-size, paragraph-based
* Embedding engines: OpenAI small & large models
* Modular pipeline for ingestion → chunking → embedding → storage → retrieval → reranking
* Streamlit UI for interactive querying
* FastAPI backend for programmatic access
* Evaluation framework with Excel logging

### System Modules

* **Ingestion**: `core/ingest.py` 
* **Chunking**: `core/chunking.py` 
* **Embeddings**: `core/embeddings.py` 
* **Vector Databases**: `core/vectordbs.py` 
* **Retrieval Engine**: `core/retrievers.py` 
* **Reranking Strategies**: `core/rerankers.py` 
* **RAG Pipeline**: `core/rag_pipeline.py` 
* **Evaluation Metrics**: `metrics/eval_metrics.py`
* **Excel Logging**: `metrics/loggig_excel.py`
* **FastAPI Backend**: `api/main.py` 

---

## Folder Structure

```
RagLab/
│
├── .env                     # Environment variables (not committed)
├── requirements.txt         # Python dependencies
│
├── api/                     # FastAPI service
│   └── main.py
│
├── core/                    # Main RAG pipeline modules
│   ├── chunking.py
│   ├── embeddings.py
│   ├── ingest.py
│   ├── rag_pipeline.py
│   ├── retrievers.py
│   ├── rerankers.py
│   └── vectordbs.py
│
├── config/
│   └── settings.py
│
├── ui/                      # Streamlit frontend
│
├── logs/                    # Excel logs for queries and evaluation
│
├── metrics/
│   ├── loggig_excel.py
│   └── eval_metrics.py
│
├── store/                   # ChromaDB persistence
├── data/                    # Demo files, ingestion sources, eval_set.json
└── assets/                  # UI icons
```

---

## Installation

### Requirements

Python 3.10+

### Install Dependencies

```
pip install -r requirements.txt
```

Or, if using Poetry:

```
poetry install
```

---

## Environment Variables

Create a `.env` file at the project root:

```
OPENAI_API_KEY=your_key_here
LLM_MODEL_OPENAI=gpt-4o-mini

EMBEDDING_MODEL=text-embedding-3-small

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

QDRANT_COLLECTION_PREFIX=raglab_
```

---

## Ingestion Workflow

RagLab ingests files into the selected vector DB using:

`core/ingest.py` 

Example:

```python
from core.ingest import ingest_file

ingest_file(
    filename="demo.txt",
    collection_name="demo",
    db_type="chroma",
    chunker_name="fixed",
    chunk_size=1000,
    chunk_overlap=200,
    embedding_backend="openai_small",
)
```

---

## Retrieval + RAG Pipeline

Core pipeline: `core/rag_pipeline.py` 

Steps:

1. Embed query
2. Retrieve top-k from selected vector DB
3. Apply optional reranker
4. Build context
5. LLM answers strictly using retrieved context

Example:

```python
from core.rag_pipeline import answer_question

result = answer_question(
    question="What is this document about?",
    collection_name="demo",
    k=4,
    db_type="chroma",
    embedding_backend="openai_small",
    reranker="llm",
)
print(result["answer"])
```

---

## FastAPI Usage

The backend exposes `/query`:

From `api/main.py` :

### POST /query

**Request Body:**

```json
{
  "question": "your question",
  "collection_name": "demo",
  "k": 4
}
```

**Response:**

```json
{
  "question": "...",
  "answer": "..."
}
```

Run server:

```
uvicorn api.main:app --reload
```

---

## Evaluation

Evaluation uses `metrics/eval_metrics.py`, reading from `data/eval_set.json`:

Run:

```
python metrics/eval_metrics.py
```

Outputs:

* contains-match accuracy
* token-overlap score
* rows logged to Excel

Example eval set: `data/eval_set.json` 

---

## Notes

* Chroma stores vectors in `store/chroma/`
* Qdrant requires correct `vector_size` matching embedding model
* Reranking improves contextual relevance
* Paragraph chunking improves semantic grouping for long texts

---

## License

standard MIT license
