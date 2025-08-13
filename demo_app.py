import re
import streamlit as st
from pathlib import Path

from rag_core.ingest import ingest
from rag_core.qa import load_index, answer_and_summarize, load_conv_chain
from textwrap import dedent
from pathlib import Path
from io import BytesIO
from openai import OpenAI

import os
import tempfile            # <-- add this
from uuid import uuid4     # <-- only needed if you keep/use speak_answer()
import html



# 1) Hard-code your key (DO NOT COMMIT THIS)
os.environ["OPENAI_API_KEY"] = "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"

# 2) Create the OpenAI client using that key
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])




def speak_answer(answer_text: str):
    if not answer_text:
        return

    client = OpenAI()
    out_path = Path(f"/tmp/tts_{uuid4().hex}.mp3")

    try:
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=answer_text,
        ) as resp:
            resp.stream_to_file(out_path)

        with open(out_path, "rb") as f:
            st.audio(f.read(), format="audio/mp3")
    except Exception as e:
        st.warning(f"TTS error: {e}")




@st.cache_data(show_spinner=False)
def tts_openai_bytes(
    text: str,
    model: str = "gpt-4o-mini-tts",
    voice: str = "alloy",
) -> bytes:
    """Return MP3 bytes for the given text using OpenAI TTS."""
    if not text:
        return b""

    try:
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp_path = tmp.name

        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
        ) as resp:
            resp.stream_to_file(tmp_path)

        with open(tmp_path, "rb") as f:
            data = f.read()
        os.remove(tmp_path)
        return data
    except Exception as e:
        st.warning(f"TTS error: {e}")
        return b""



st.set_page_config(page_title="PipelineIQ", page_icon="/Users/mousa/Desktop/PipelineIQ/assets/adnoc-logo2.png", layout="wide")

CONTENT_WIDTH = 1100  # pick one number and everything will match it

st.markdown(
    f"""
    <style>
      :root {{
        --adnoc-blue: #0047BA;
        --bg: #F0F4F8;
        --card-bg: #FFFFFF;
        --text: #111111;
        --muted: #555555;
        --border: #D9E2EC;
      }}

      /* Page */
      main {{ background-color: var(--bg); }}
      footer {{ visibility: hidden; }}
      .block-container {{ padding-top: 2rem !important; }}

      /* One wrapper that controls the width of EVERYTHING */
      .piq-container {{
        max-width: {CONTENT_WIDTH}px;
        margin: 0 auto;
      }}

      /* Buttons */
      .stButton > button {{
        background-color: var(--adnoc-blue) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1rem !important;
        font-weight: 600 !important;
      }}
      
      /* Form container styling */
      .stForm {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 24px 28px;
        margin-top: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
      }}

      /* Cards share the same width (100% of the wrapper) */
      .chat-card {{
        max-width: 100%;
        box-sizing: border-box;
      }}

      .chat-card {{
        background: var(--card-bg);
        border-left: 4px solid var(--adnoc-blue);
        border-radius: 10px;
        padding: 18px 20px 16px 20px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);

        /* allow top-right chips box */
        position: relative;
      }}

      .chat-card h4 {{
        margin: 0 0 6px 0;
        color: var(--adnoc-blue);
        font-size: 1.05rem;
      }}

      .chat-card p {{
        margin: 0 0 10px 0;
        line-height: 1.55;
        color: #111;
      }}

      .chat-card ul {{
        margin: 0.4rem 0 0.6rem 1.1rem;
        color: #333;
      }}

      /* top-right box for sources */
      .src-box {{
        position: absolute;
        top: 10px;
        right: 12px;
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
        justify-content: flex-end;
        max-width: 55%;
      }}

      .src-chip {{
        display: inline-block;
        background: #E6F0FF;
        color: #0047BA;
        padding: 2px 8px;
        margin: 0;               /* chips are inside a flex box now */
        border-radius: 999px;
        font-size: 0.82rem;
        border: 1px solid #CCE0FF;
        white-space: nowrap;
      }}

      .meta-line {{
        color: #7A869A;
        font-size: 0.78rem;
        margin-top: 6px;
      }}

      /* Action toolbar that sits on the card's top-right corner */
      .card-toolbar {{
        position: relative;
        margin-top: -36px;       /* pulls the toolbar up onto the card */
        margin-right: 6px;
        display: flex;
        justify-content: flex-end;
        z-index: 2;
      }}

      .card-toolbar .stButton > button {{
        background: #E6F0FF !important;
        color: #0047BA !important;
        border: 1px solid #CCE0FF !important;
        border-radius: 999px !important;
        padding: 4px 10px !important;
        font-weight: 600 !important;
      }}

      /* audio inline spacing so it feels part of the card */
      .audio-inline {{
        margin-top: 8px;
      }}

      

      /* A little footer area that visually sits INSIDE the card, under the bullets */
      .card-footer {{
        margin: 8px 20px 6px 20px;   /* matches the card’s inner padding */
      }}

      .card-footer .stButton > button {{
        background: #E6F0FF !important;
        color: #0047BA !important;
        border: 1px solid #CCE0FF !important;
        border-radius: 999px !important;
        padding: 4px 12px !important;
        font-weight: 600 !important;
      }}

      /* tighter spacing for the inline audio player */
      .card-audio {{
        margin-top: 6px;
      }}

    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────
# 2) HEADER (logo + title)
# ─────────────────────────────
logo_path = Path(__file__).parent / "assets" / "adnoc_logo.png"

col_logo, col_title = st.columns([1, 8], gap="small")
with col_logo:
    st.image(str(logo_path), width=88)
with col_title:
    st.markdown(
        """
        <h1 style="margin:0;color:#0047BA;font-size:2.6rem;font-weight:800;">
          PipelineIQ
        </h1>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────
# Sidebar
# ─────────────────────────────
with st.sidebar:
    st.header("Admin")
    if st.button("Re-Ingest PDFs", use_container_width=True):
        ingest()
        st.success("Re-ingestion complete!")

    st.divider()

    if st.button("Clear chat history", use_container_width=True):
        # wipe just what we use
        st.session_state.pop("history", None)
        st.session_state.pop("conv_chain", None)   # optional: force a fresh chain next run
        st.session_state.pop("ask_input", None)

        # Streamlit >= 1.27
        if hasattr(st, "rerun"):
            st.rerun()
        else:  # older Streamlit
            st.experimental_rerun()

# ─────────────────────────────
# State init
# ─────────────────────────────
if "conv_chain" not in st.session_state:
    st.session_state.conv_chain = load_conv_chain()

if "history" not in st.session_state:
    st.session_state.history = []  # list of (q, a, summary)

# Cache the vectorstore only once per run to avoid reload every question
@st.cache_resource(show_spinner=False)
def _get_index():
    return load_index()







def chips_from_docs(docs):
    """
    Build labels like 'file.pdf (page: 12, 13)' from retrieved docs.
    Handles LangChain Documents, dicts, or plain strings.
    """
    pages_by_src = {}

    for d in docs or []:
        # 1) Document with metadata
        if hasattr(d, "metadata"):
            src = str(d.metadata.get("source", ""))
            page = d.metadata.get("page", None)
        # 2) dict-like
        elif isinstance(d, dict):
            src = str(d.get("source", ""))
            page = d.get("page", None)
        # 3) plain string (just a source)
        else:
            src = str(d)
            page = None

        if not src:
            continue

        src_name = Path(src).name
        pages_by_src.setdefault(src_name, set())
        if isinstance(page, int):
            pages_by_src[src_name].add(page)

    labels = []
    for src_name, pages in pages_by_src.items():
        if not pages:
            labels.append(src_name)
        else:
            p = sorted(pages)
            if len(p) == 1:
                labels.append(f"{src_name} (page: {p[0]})")
            else:
                labels.append(f"{src_name} (pages: {', '.join(map(str, p))})")
    return labels


def render_card(q: str, a: str, s: str, docs, key: str | None = None) -> None:
    """
    Render one Q/A card with:
      - source chips (with page numbers)
      - Q and A text
      - 3-bullet summary 's'
      - a 'Listen' (TTS) button visually inside the card under the bullets

    Requires helper functions:
      - chips_from_docs(docs) -> list[str]
      - tts_openai_bytes(text) -> bytes | None
    """
    # Ensure each card/widgets get a unique key
    if key is None:
        key = uuid4().hex

    # Build chips
    chip_labels = chips_from_docs(docs) if docs else []
    chips_html = "".join(f"<span class='src-chip'>{html.escape(lbl)}</span>" for lbl in chip_labels)

    # Build bullets from the summary 's'
    bullets_html = ""
    if s:
        items = []
        for line in s.split("\n"):
            t = line.strip()
            if t.startswith("-"):
                items.append(f"<li>{html.escape(t.lstrip('- ').strip())}</li>")
        if items:
            bullets_html = "<ul>" + "".join(items) + "</ul>"

    # Render the card shell (note: pure HTML — Streamlit widgets go next)
    st.markdown(
        f"""
        <div class="chat-card" id="card-{key}">
          <div class="src-box">{chips_html}</div>
          <p><strong style="color:#0047BA;">Q:</strong> {html.escape(q)}</p>
          <p><strong style="color:#0047BA;">A:</strong> {a}</p>
          {bullets_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add a small widget row that *looks* inside the card.
    # We nudge it up with CSS so it visually sits within the card, under the bullets.
    # (Streamlit can’t truly place widgets inside raw HTML, so we mimic it.)
    with st.container():
        c1, c2 = st.columns([0.18, 1], vertical_alignment="center")
        with c1:
            play = st.button("🔊 Listen", key=f"tts_btn_{key}", help="Read this answer aloud")
        with c2:
            if play:
                audio = tts_openai_bytes(a)
                if audio:
                    st.audio(audio, format="audio/mp3")
                else:
                    st.info("Could not generate audio.")

    # Nudge the widget row up so it visually belongs to the card
    st.markdown(
        f"""
        <style>
          /* The container Streamlit renders right after the card gets pulled up */
          #card-{key} + div.stContainer {{
            margin-top: -10px;     /* pull it into the card */
            margin-left: 8px;
            margin-right: 8px;
            padding-bottom: 6px;
          }}
          /* Make the Listen button a bit smaller so it feels like a card action */
          [data-testid="baseButton-secondary"][data-testid][aria-label="🔊 Listen"],
          .stButton > button[kind="secondary"] {{
            padding: 0.25rem 0.6rem;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

index = _get_index()

# ─────────────────────────────
# Ask form
# ─────────────────────────────
st.markdown(
    '<h2 style="margin-top:1rem; margin-bottom:0; color:#0047BA;">Ask a question</h2>',
    unsafe_allow_html=True,
)

with st.form("ask_form", clear_on_submit=True):
    user_query = st.text_input(
        "Your query",
        placeholder="e.g. What does a blowout preventer do?",
        label_visibility="collapsed",
    )
    ask = st.form_submit_button("Ask")

# ─────────────────────────────
# Helper: extract sources from answer tail
# ─────────────────────────────
def extract_sources_and_clean(answer_text: str):
    """
    Parse 'SOURCES: ...' from the answer (anywhere in the string).
    Returns (clean_answer_without_sources, [sources]).
    """
    if not answer_text:
        return answer_text, []

    # Find the first SOURCES: occurrence (case-insensitive), grab everything after it
    m = re.search(r"(?i)\bSOURCES?:\s*(.+)$", answer_text.strip(), flags=re.DOTALL)
    if not m:
        return answer_text.strip(), []

    # Text after 'SOURCES:' (may include multiple filenames separated by commas/semicolons/bullets/newlines)
    src_tail = m.group(1).strip()

    # Clean: remove the SOURCES: … part from the answer body
    clean = re.sub(r"(?i)\s*SOURCES?:\s*.+$", "", answer_text, flags=re.DOTALL).strip()

    # Split on common separators
    parts = re.split(r"[;,•\n]\s*", src_tail)
    seen, sources = set(), []
    for p in parts:
        t = p.strip()
        if t and t not in seen:
            seen.add(t)
            sources.append(t)
    return clean, sources

# ─────────────────────────────
# Handle submit
# ─────────────────────────────
# when the user submits:
if ask and user_query:
    with st.spinner("Thinking…"):
        if st.session_state.history:
            # Conversational turn
            resp = st.session_state.conv_chain({
                "question": user_query,
                "chat_history": [(q, a) for q, a, *_ in st.session_state.history],
            })
            answer = resp["answer"]
            docs = resp.get("source_documents", [])  # <- Document list
            # Summarize the answer text (cheap)
            _, summary, _ = answer_and_summarize(answer, index)
        else:
            # First turn: do RAG directly and keep the docs we retrieved
            answer, summary, docs = answer_and_summarize(user_query, index)

        # Save docs with the turn
        st.session_state.history.append((user_query, answer, summary, docs))

# ─────────────────────────────
# Render history (latest first)
# ─────────────────────────────
from textwrap import dedent

# Render newest first
# Render newest first, with unique keys
for idx, item in enumerate(reversed(st.session_state.history)):
    if len(item) == 4:
        q, a, s, docs = item
    else:
        q, a, s = item
        docs = []
    render_card(q, a, s, docs, key=f"card-{idx}")