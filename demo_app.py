import re
import streamlit as st
from pathlib import Path

from rag_core.ingest import ingest
from rag_core.qa import load_index, answer_and_summarize, load_conv_chain
from textwrap import dedent

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
if ask and user_query:
    with st.spinner("Thinking…"):
        if st.session_state.history:
            # Conversational QA
            resp = st.session_state.conv_chain({
                "question": user_query,
                "chat_history": [(q, a) for q, a, _ in st.session_state.history],
            })
            answer = resp["answer"]
            # Summarize the *answer text* (cheaper) or the whole context if you prefer
            _, summary = answer_and_summarize(answer, index)
        else:
            # First turn: simple RAG
            answer, summary = answer_and_summarize(user_query, index)

        st.session_state.history.append((user_query, answer, summary))

# ─────────────────────────────
# Render history (latest first)
# ─────────────────────────────
for q, a, s in reversed(st.session_state.history):
    clean_answer, sources = extract_sources_and_clean(a)

    bullets = "".join(
        f"<li>{line.strip('- ').strip()}</li>"
        for line in s.split("\n")
        if line.strip().startswith("-")
    )
    chips = "".join(f"<span class='src-chip'>{src}</span>" for src in sources)

    card_html = dedent(f"""
    <div class="chat-card">
      <div class="src-box">{chips}</div>
      <p><strong style="color:#0047BA;">Q:</strong> {q}</p>
      <p><strong style="color:#0047BA;">A:</strong> {clean_answer}</p>
      {('<ul>' + bullets + '</ul>') if bullets else ''}
    </div>
    """).strip()

    st.markdown(card_html, unsafe_allow_html=True)