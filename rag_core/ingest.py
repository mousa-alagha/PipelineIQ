# rag_core/ingest.py
from __future__ import annotations

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pypdf import PdfReader  # pip install pypdf
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS


def ingest(data_dir: str = "data/raw_documents", store_dir: str = "vectorstore") -> None:
    """
    Read PDFs, create one Document per *page* with metadata
    {"source": <filename>, "page": <1-based>}, split into chunks, and save FAISS.
    """
    load_dotenv()

    docs: List[Document] = []
    data_path = Path(data_dir)

    for pdf_path in sorted(data_path.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf_path.name, "page": i},
                )
            )

    if not docs:
        raise RuntimeError("No pages with text were ingested.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunked = splitter.split_documents(docs)  # keeps metadata (source/page)

    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_documents(chunked, embeddings)
    vs.save_local(store_dir)
    print(f"[ingest] Saved {len(chunked)} chunks to {store_dir}")