import os, json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PyPDF2.errors import DependencyError
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

def ingest(
    data_dir: str = "data/raw_documents",
    meta_path: str = "data/metadata.json",
    store_dir: str = "vectorstore",
):
    """Read PDFs, extract & chunk text, embed, and save FAISS index."""
    # Load environment (OPENAI_API_KEY)
    load_dotenv()

    # 1) Load metadata file with filenames and titles
    with open(meta_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    # 2) Initialize text splitter and containers
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts, metadatas = [], []

    # 3) Process each document
    for doc in docs:
        filename = doc["filename"]
        file_path = os.path.join(data_dir, filename)
        try:
            reader = PdfReader(file_path)
        except DependencyError:
            print(f"⛔ Skipping encrypted file: {filename}")
            continue

        # Extract full text and split into chunks
        full_text = "".join(page.extract_text() or "" for page in reader.pages)
        chunks = splitter.split_text(full_text)

        # Append to lists
        texts.extend(chunks)
        metadatas.extend([
            {"title": doc["title"], "source": filename}
            for _ in chunks
        ])

    print(f"> Prepared {len(texts)} chunks from {len(docs)} documents.")

    # 4) Build embeddings and FAISS index
    embeddings = OpenAIEmbeddings()
    index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)


    os.makedirs(store_dir, exist_ok=True)
    index.save_local(store_dir)
    print(f"> Vector store saved to: {store_dir}")