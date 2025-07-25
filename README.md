# PipelineIQ

**AI‐powered semantic search & QA for oil & gas documentation**

## Setup

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` → `.env` and add your `OPENAI_API_KEY`
4. Put your PDFs in `data/raw_documents/` and update `data/metadata.json`
5. `jupyter lab notebooks/1_ingest.ipynb` to build your vectorstore
6. `streamlit run demo_app.py` to launch the demo UI