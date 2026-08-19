# Setup: LangChain + OpenAI + Chroma

This document explains how to set up the environment to use the LangChain-backed RAG pipeline included in `src/comp_agent/langchain_rag.py`.

1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```bash
pip install -e .
# or install required extras directly
pip install langchain openai chromadb sentence-transformers biopython
```

3) Set OpenAI API key (if using OpenAI embeddings/LLM)

```bash
export OPENAI_API_KEY="sk-..."
```

4) Ingest documents

Use `LangChainRAG.ingest_texts([...], metadatas=[...])` to add documents. For PubMed abstracts you can fetch via `Bio.Entrez` from `biopython` and ingest the abstracts.

5) Query and answer

```python
from comp_agent.langchain_rag import LangChainRAG

pc = LangChainRAG(embedding_type="openai", persist_dir="chroma_db")
pc.ingest_texts(["abstract text ..."], metadatas=[{"source": "pubmed:12345"}])
print(pc.answer("What does this paper say about X?"))
```

Notes
- Chroma will persist embeddings to `persist_dir`.
- Replace `OpenAI` LLM with another LLM adapter supported by LangChain if desired.
- For larger-scale, switch vector store to PostgreSQL+pgvector or Milvus.
