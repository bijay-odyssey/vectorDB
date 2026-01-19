# VectorDB – Qdrant Semantic Search & RAG Demo

This repository demonstrates how to build a **simple Vector Database–powered semantic search system** using **Qdrant** and **Sentence Transformers**.
It showcases **document ingestion, vector similarity search, metadata filtering**, and an **interactive query interface**, forming the foundation of a **Retrieval-Augmented Generation (RAG)** pipeline.

---

##  Features

* Vector embedding generation using **Sentence Transformers**
* Vector storage and similarity search using **Qdrant**
* Metadata-based filtering (e.g., category)
* Simple semantic search demo
* Interactive command-line search with optional filters
* Lightweight and easy to extend for full RAG pipelines

---

##  Tech Stack

* **Python 3.8+**
* **Qdrant** (Vector Database)
* **Sentence Transformers**
* **all-MiniLM-L6-v2** embedding model

---

##  Project Structure

```text
vectorDB/
│
├── rag_qdrant.py
│   ├── Ingests documents into Qdrant
│   ├── Performs semantic search
│   └── Performs metadata-filtered search
│
├── rag_qdrant_input.py
│   ├── Ingests documents into Qdrant
│   └── Interactive CLI-based semantic search
│
└── README.md
```

---

##  Getting Started

### 1️ Prerequisites

Make sure you have:

* Python 3.8 or higher
* Docker (recommended for running Qdrant)

---

### 2️ Run Qdrant

Start Qdrant locally using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Qdrant will be available at:

```
http://localhost:6333
```

---

### 3️ Install Dependencies

```bash
pip install qdrant-client sentence-transformers
```

---

##  Example Documents

The demo uses sample documents with metadata:

```json
{
  "title": "Vector Databases",
  "text": "Vector databases store embeddings of text for semantic search.",
  "category": "database"
}
```

Each document is:

* Converted into an embedding
* Stored in Qdrant
* Queried using vector similarity

---

##  Running the Scripts

---

###  Option 1: Run Basic Semantic & Filtered Search

```bash
python rag_qdrant.py
```

**What it does:**

* Creates a Qdrant collection (if not exists)
* Ingests documents with metadata
* Runs:

  * Semantic search
  * Category-filtered search

**Example Output:**

```text
Search Results for: 'How do vector databases work?'
- Vector Databases | score=0.82 | category=database
```

---

###  Option 2: Run Interactive Search (Recommended)

```bash
python rag_qdrant_input.py
```

You’ll get an interactive prompt:

```text
Your query:
```

#### Example Queries

* Simple semantic search:

```text
vector database
```

* Semantic search with category filter:

```text
vector database |category:database
```

* AI-related query:

```text
RAG systems |category:AI
```

---

##  How Filtering Works

Filtering is done using **payload metadata** stored in Qdrant.

Example filter logic:

* `category: database`
* `category: AI`

This allows combining **semantic similarity + structured filtering**, a key concept in RAG systems.

---

##  Use Cases

* Semantic document search
* Metadata-aware search systems
* RAG (Retrieval-Augmented Generation) pipelines
* AI-powered knowledge bases
* Search engines with embeddings

---

##  Future Improvements

* Add LLM integration (OpenAI / HuggingFace)
* Support document ingestion from files (PDF, TXT)
* Add hybrid search (keyword + vector)
* REST API using FastAPI
* Persistent storage for embeddings

---

##  License

This project is open-source and available under the **MIT License**.

---

##  Author

**Bijay**
GitHub: [bijay-odyssey](https://github.com/bijay-odyssey)

---
 
