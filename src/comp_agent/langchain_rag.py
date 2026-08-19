from typing import List, Optional

try:
    from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.docstore.document import Document as LCDocument
    from langchain.chains import RetrievalQA
    from langchain.llms import OpenAI
except Exception:  # pragma: no cover - optional imports
    OpenAIEmbeddings = None
    HuggingFaceEmbeddings = None
    Chroma = None
    RecursiveCharacterTextSplitter = None
    LCDocument = None
    RetrievalQA = None
    OpenAI = None


class LangChainRAG:
    """A small adapter to build a LangChain-based RAG pipeline using Chroma.

    Usage:
      pipeline = LangChainRAG(embedding_type="openai")
      pipeline.ingest_texts(["text ..."], metadatas=[{"source": "file1"}])
      answer = pipeline.answer("What is X?")
    """

    def __init__(self, persist_dir: str = "chroma_db", embedding_type: str = "openai", hf_model: str = "all-MiniLM-L6-v2"):
        if OpenAIEmbeddings is None:
            raise RuntimeError("LangChain or its dependencies are not installed. Install via pyproject or pip.")
        self.persist_dir = persist_dir
        self.embedding_type = embedding_type
        if embedding_type == "openai":
            self.embedder = OpenAIEmbeddings()
        else:
            # Use HuggingFace local embeddings (sentence-transformers)
            self.embedder = HuggingFaceEmbeddings(model_name=hf_model)

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        self._client = None

    def _get_chroma(self, texts: Optional[List[str]] = None, metadatas: Optional[List[dict]] = None):
        if Chroma is None:
            raise RuntimeError("Chroma/Chromadb not installed")
        if texts is None:
            # load existing collection
            return Chroma(persist_directory=self.persist_dir, embedding_function=self.embedder)
        return Chroma.from_texts(texts=texts, embedding=self.embedder, metadatas=metadatas, persist_directory=self.persist_dir)

    def ingest_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        # split texts into smaller docs
        docs = []
        metas = []
        for i, t in enumerate(texts):
            chunks = self.text_splitter.split_text(t)
            for j, c in enumerate(chunks):
                docs.append(c)
                metas.append(dict((metadatas[i] if metadatas else {}) , **{"chunk_index": j}))

        col = self._get_chroma(texts=docs, metadatas=metas)
        col.persist()
        self._client = col
        return len(docs)

    def get_retriever(self, k: int = 4):
        if self._client is None:
            self._client = self._get_chroma()
        return self._client.as_retriever(search_kwargs={"k": k})

    def answer(self, query: str, k: int = 4):
        if RetrievalQA is None or OpenAI is None:
            raise RuntimeError("LangChain LLM or RetrievalQA not available; check installation.")
        retriever = self.get_retriever(k=k)
        llm = OpenAI()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        return qa.run(query)
