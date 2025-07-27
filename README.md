# PipelineIQ

**AI-powered Retrieval-Augmented Generation (RAG) over oil & gas PDFs** — with a clean, ADNOC-branded Streamlit UI.  
PipelineIQ ingests your PDFs, chunks and embeds them with OpenAI, stores vectors in FAISS, and lets you chat with your corpus with sources + summaries.

---

## Features

- **PDF → chunks → OpenAI embeddings → FAISS vector store**
- **Conversational QA** (keeps your chat history in-session)
- **Cited answers + 3-bullet summaries**
- **One-click “Re-Ingest PDFs”** button in the UI
- **ADNOC-themed** (colors + logo) Streamlit interface
- Clear **project structure**: `rag_core/` holds ingestion & QA logic, UI stays thin

---

## Architecture (high level)



```mermaid
sequenceDiagram
  autonumber
  participant U as User
  participant UI as Streamlit UI
  participant R as Router
  participant RAG as RAG
  participant H as Hybrid
  participant VS as FAISS/BM25
  participant L as LLM
  participant G as Guard-Summarizer
  participant ING as Ingestion Pipeline

  U->>UI: Enter question
  UI->>R: Query + history
  R->>RAG: Retrieve context
  RAG->>H: Hybrid search
  H->>VS: Semantic + keyword lookup
  VS-->>H: Top-k chunks
  H-->>RAG: Reranked chunks
  R->>L: Prompt + chunks
  L-->>R: Draft answer
  R->>G: Check citations + summarize
  G-->>R: Verified answer + 3 bullets
  R-->>UI: Answer + sources + bullets

  opt Admin - clear history
    UI->>R: Clear history
    R-->>UI: History cleared
  end

  opt Admin - re-ingest PDFs
    UI->>R: Re-ingest command
    R->>ING: Chunk + embed documents
    ING->>VS: Update vector store(s)
    R-->>UI: Re-ingestion complete
  end


1. **Ingestion (`rag_core/ingest.py`)**
   - Read `data/metadata.json`
   - Extract text from `data/raw_documents/*.pdf`
   - Chunk with `RecursiveCharacterTextSplitter`
   - Embed with **OpenAIEmbeddings**
   - Store in **FAISS** (`vectorstore/`)

2. **Querying (`rag_core/qa.py`)**
   - Load FAISS, retrieve top-k chunks
   - Run **LLM QA** chain to generate an answer with sources
   - Produce a **3-bullet summary**

3. **UI (`demo_app.py`)**
   - Streamlit app with ADNOC branding
   - Ask questions, see history, re-ingest PDFs from sidebar

---

## Repository Layout

```
PipelineIQ/
├── data/
│   ├── raw_documents/           # your PDFs go here
│   └── metadata.json            # [{"title": "...", "filename": "..."}]
├── notebooks/
│   └── 1_ingest.ipynb           # optional: ingest via notebook
├── rag_core/
│   ├── __init__.py
│   ├── ingest.py                # build & persist FAISS vector store
│   └── qa.py                    # load index, QA & summarization helpers
├── vectorstore/                 # generated FAISS artifacts (after ingest)
├── .streamlit/
│   └── config.toml              # Streamlit theme (ADNOC colors)
├── .env.example                 # copy to .env and set OPENAI_API_KEY
├── demo_app.py                  # Streamlit UI
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python **3.10+**
- An **OpenAI API key**

---

## Setup

### 1) Create & activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment

```bash
cp .env.example .env
# open .env and set:
# OPENAI_API_KEY=sk-...
```

### 4) Add your PDFs

- Put your files in: `data/raw_documents/`
- List them in `data/metadata.json`, e.g.:

```json
[
  { "title": "Oil & Gas Production Handbook", "filename": "oil_gas_handbook.pdf" },
  { "title": "Blowout Preventers", "filename": "Blowout-Preventers-1.pdf" }
]
```

### 5) Build the vector store (pick one)

**A) From the Streamlit UI (easiest)**

```bash
streamlit run demo_app.py
```

Open the app → click **Re-Ingest PDFs** in the sidebar.

**B) From CLI**

```bash
python -m rag_core.ingest
```

**C) From the notebook**

```bash
jupyter lab notebooks/1_ingest.ipynb
```

### 6) Run the app

```bash
streamlit run demo_app.py
```

It will start on http://localhost:8501.

---

## Usage

1. Type a question in the input box.
2. The app retrieves the most similar chunks from FAISS.
3. The LLM answers and returns:
   - **Answer**
   - **3-bullet summary**
   - **Sources** (PDF + page)
4. Your past Q&A stays on the page (session memory).
5. Click **Clear chat history** to wipe the session.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'tiktoken'`

```bash
pip install tiktoken
```

### `ValidationError: Did not find openai_api_key`

Ensure `.env` exists and `OPENAI_API_KEY` is set, then:

```bash
source .venv/bin/activate
export OPENAI_API_KEY=sk-...   # or rely on python-dotenv loading .env
```

### `DependencyError: PyCryptodome is required for AES algorithm`

```bash
pip install pycryptodome
```

### FAISS “dangerous deserialization” error

When loading your **own** FAISS store:

```python
FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
```

(Do **not** enable this if the index comes from an untrusted source.)

---

## Roadmap (ideas)

- Swap FAISS → **Chroma** or **PGVector**
- Add **Rerankers** (e.g., Cohere / Cross-encoders)
- **Multi-modal** support (images, tables)
- **Agents / Tools** (diagnostics, troubleshooting steps)
- **Access control & user auth**
- **Docker** for easy deployment

---

## License

Proprietary / Internal (adjust to your needs).
