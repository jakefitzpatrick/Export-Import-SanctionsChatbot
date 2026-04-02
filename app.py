import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import openai
import plotly.graph_objects as go
import streamlit as st

from rag import (
    DEFAULT_EMBEDDING_MODEL,
    SOURCE_KIND_PDF,
    RagIndex,
    create_document_source,
)

load_dotenv()

DEFAULT_TOP_K = 3
DEFAULT_CHUNK_SIZE = 250
DEFAULT_CHUNK_OVERLAP = 50

CHUNK_CONFIG_BY_KIND = {
    SOURCE_KIND_PDF: {"chunk_size": 200, "chunk_overlap": 100},
}

QUESTION_PLACEHOLDER = (
    "Ask about OFAC SDN Enhanced or U.S. HTS topics (provide enough detail for retrieval):"
)

COMBINED_SYSTEM_PROMPT = (
    'You are a compliance assistant that relies purely on the provided PDF. '
    'Always ground your answers in the retrieved evidence below, cite each chunk with its source label, chunk number, and page number (e.g., "(Source finalCopy_2026HTSRev4.pdf | Source 3 | Page 5)"). '
    'Prefer exact matches to the user query terms; if any necessary context is missing from the retrieved pages, explicitly flag which pages or sections are unavailable before offering general guidance. '
    'Do not hallucinate facts; begin every response with "Answer:" followed by a concise conclusion.'
)

GENERATION_NO_CONTEXT_MESSAGE = (
    "The retriever did not return any snippets from any source, so be transparent that no direct hits were found and answer based on the documented scope of the datasets."
)

RESPONSE_STRUCTURE_INSTRUCTION = (
    'Structure your response in two parts: first a concise "Answer:" line that directly addresses the user question, then a "Supporting Evidence:" section with short bullet(s) referencing the chunk(s) you used. '
    'Each bullet must end with the citation in the format (Source <label> | Source <chunk number>) so users know exactly which chunk resolved the question.'
)

COUNTRY_RISK = {
    "Cameroon": {"score": 73.79, "year": 2025},
    "Russia": {"score": 91.2, "year": 2025},
    "China": {"score": 65.4, "year": 2025},
    "Iran": {"score": 95.1, "year": 2025},
    "Germany": {"score": 12.3, "year": 2025},
    "Canada": {"score": 8.7, "year": 2025},
    "Brazil": {"score": 48.5, "year": 2025},
    "Nigeria": {"score": 78.2, "year": 2025},
    "France": {"score": 15.1, "year": 2025},
    "India": {"score": 52.3, "year": 2025},
}


def get_risk_color(score: float) -> tuple[str, str]:
    if score >= 75:
        return "#c0392b", "High"
    if score >= 45:
        return "#d35400", "Medium"
    return "#1e8449", "Low"


@st.cache_resource(show_spinner=False)
def load_rag_index(
    embedding_model: str,
    source_kind: str,
    source_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RagIndex:
    document_source = create_document_source(source_kind, Path(source_path))
    index = RagIndex(
        document_source=document_source,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    index.ensure_index(embedding_model)
    return index


def _format_reference_sections(results: list[dict], source_kind: str, source_label: str) -> str:
    lines: list[str] = []
    for idx, item in enumerate(results, start=1):
        chunk_index = item.get("chunk_index")
        chunk_number = chunk_index + 1 if chunk_index is not None else idx
        metadata = item.get("metadata", {}) or {}
        page = metadata.get("page")
        page_part = f" | Page {page}" if page else ""
        citation = f"(Source {source_label} | Source {chunk_number}{page_part})"
        lines.append(
            f"{citation}\nScore: {item.get('score', 0):.3f}\n{item.get('chunk', '')}"
        )
    return "\n\n".join(lines)


def build_generation_messages(contexts: list[tuple[str, str]], question: str) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": COMBINED_SYSTEM_PROMPT}]
    if contexts:
        for label, context_text in contexts:
            messages.append(
                {
                    "role": "system",
                    "content": f"Retrieved context from {label}:\n{context_text}",
                }
            )
    else:
        messages.append({"role": "system", "content": GENERATION_NO_CONTEXT_MESSAGE})
    messages.append({"role": "system", "content": RESPONSE_STRUCTURE_INSTRUCTION})
    messages.append({"role": "user", "content": question})
    return messages


def main() -> None:
    st.set_page_config(page_title="ImportInsight AI", layout="wide")

    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] {
        background-color: #0f1f38;
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] * { color: #e8edf5 !important; }
    [data-testid="stSidebar"] h3 {
        color: #a0aec0 !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .bubble-user {
        background-color: #1a3a5c;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 4px 0;
        max-width: 70%;
        float: right;
        clear: both;
        font-size: 14px;
        line-height: 1.5;
    }
    .bubble-bot {
        background-color: #ffffff;
        color: #1a1a2e;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 4px 0;
        max-width: 70%;
        float: left;
        clear: both;
        font-size: 14px;
        line-height: 1.5;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .bubble-label {
        font-size: 10px;
        color: #94a3b8;
        margin-bottom: 2px;
        clear: both;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .bubble-label-right { text-align: right; }
    .timestamp {
        font-size: 10px;
        color: #cbd5e1;
        margin-top: 3px;
        clear: both;
    }
    .timestamp-right { text-align: right; }
    .clearfix { clear: both; }
    img { mix-blend-mode: multiply; }
    [data-testid="stSidebar"] img {
        display: block;
        margin-left: 0 !important;
        padding-left: 0 !important;
        mix-blend-mode: normal !important;
        filter: brightness(0) invert(1);
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        font-size: 14px;
    }
    .stButton > button {
        background-color: #1a3a5c;
        color: white !important;
        border-radius: 10px;
        font-weight: 500;
        border: none;
        padding: 10px;
    }
    .stButton > button:hover { background-color: #2a5298; }
    [data-testid="stHorizontalBlock"] .stButton > button {
        background-color: transparent !important;
        color: #94a3b8 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 20px !important;
        font-size: 12px !important;
        font-weight: 400 !important;
        padding: 4px 10px !important;
        opacity: 0.8;
    }
    [data-testid="stHorizontalBlock"] .stButton > button:hover {
        background-color: #f0f4ff !important;
        color: #1a3a5c !important;
        opacity: 1;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <h1 style='color:#0f1f38; font-weight:700; letter-spacing:-0.5px;'>ImportInsight AI</h1>
        <p style='color:#64748b; font-size:15px; margin-top:-10px;'>Trade and tariff intelligence powered by OFAC RAG sources.</p>
        <hr style='border: 1px solid #e2e8f0; margin-top:16px;'>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style='background-color:#fef9ec; border-left: 4px solid #f0a500; padding: 10px 16px; border-radius: 6px; margin-bottom: 20px;'>
            <span style='color:#7d5a00; font-size:13px;'>
                <b>Disclaimer:</b> This tool is for informational purposes only and does not constitute legal advice. Always consult a qualified trade compliance professional.
            </span>
        </div>
    """,
        unsafe_allow_html=True,
    )

    openai.api_type = "azure"
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if not api_key:
        st.error("Please set AZURE_OPENAI_API_KEY before running the app.")
        return
    if not openai.api_base:
        st.error("Please set AZURE_OPENAI_ENDPOINT before running the app.")
        return
    openai.api_key = api_key

    deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")
    if not deployment_id:
        st.error("Please set AZURE_OPENAI_DEPLOYMENT_ID for the chat completion deployment.")
        return

    project_root = Path(__file__).resolve().parent
    pdf_filename = "finalCopy_2026HTSRev4.pdf"
    candidates = [project_root / pdf_filename, Path.cwd() / pdf_filename]
    pdf_path = next((p for p in candidates if p.exists()), None)
    if not pdf_path:
        st.error(f"{pdf_filename} not found; place it in {project_root} or the current folder.")
        return

    sidebar_sources: list[dict] = [
        {"label": pdf_path.name, "kind": SOURCE_KIND_PDF, "path": str(pdf_path)}
    ]

    if not sidebar_sources:
        st.error("No knowledge sources were found to build the RAG index.")
        return

    source_labels = [source["label"] for source in sidebar_sources]

    with st.sidebar:
        st.image("logo.png", width=150)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Settings")
        top_k = st.slider("Chunks to consider", 1, 10, DEFAULT_TOP_K)
        page_filter_input = st.text_input(
            "Filter pages (comma-separated)",
            "",
            help="Restrict results to the numbered pages you care about.",
        )
        st.caption(
            "Searches the selected knowledge sources for semantic matches, then passes the highest scoring chunks to the chat completion."
        )
        st.caption(
            "Temperature is fixed at 1.0 because this deployment currently only supports the default value."
        )
        st.caption("Current knowledge sources: " + ", ".join(source_labels))
        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("### Governance Risk")
        selected_country = st.selectbox("Select country", list(COUNTRY_RISK.keys()))
        data = COUNTRY_RISK[selected_country]
        score = data["score"]
        year = data["year"]
        color, level = get_risk_color(score)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 28, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#a0aec0", "tickfont": {"size": 9, "color": "#a0aec0"}},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 45], "color": "#1e8449"},
                    {"range": [45, 75], "color": "#d35400"},
                    {"range": [75, 100], "color": "#c0392b"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
            title={"text": f"<b>{level} Risk</b><br><span style='font-size:11px;color:#a0aec0'>{selected_country} · {year}</span>", "font": {"size": 13, "color": "white"}},
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=60, b=10),
            height=200,
            font={"color": "white"},
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("Clear Chat", width="stretch"):
            st.session_state.messages = []
            st.rerun()
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### About")
        st.caption("ImportInsight AI helps businesses navigate trade and sanctions regulations using OFAC data.")

    try:
        rag_indexes: dict[str, RagIndex] = {}
        for source in sidebar_sources:
            chunk_settings = CHUNK_CONFIG_BY_KIND.get(source["kind"], {})
            rag_indexes[source["label"]] = load_rag_index(
                os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_ID", DEFAULT_EMBEDDING_MODEL),
                source["kind"],
                source["path"],
                chunk_size=chunk_settings.get("chunk_size", DEFAULT_CHUNK_SIZE),
                chunk_overlap=chunk_settings.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP),
            )
    except Exception as exc:
        st.error(f"Failed to prepare the RAG index: {exc}")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if len(st.session_state.messages) == 0:
        starters = [
            "Can I export software to Russia?",
            "Is Huawei on the entity list?",
            "What tariffs apply to Chinese electronics?",
            "How do I screen a foreign supplier?",
            "What goods are banned from North Korea?",
            "Are there restrictions on trading with Iran?",
        ]
        st.markdown(
            "<p style='color:#cbd5e1; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;'>Suggested questions</p>",
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, q in enumerate(starters):
            with cols[i % 2]:
                if st.button(q, key=f"starter_{i}", width="stretch"):
                    st.session_state.messages.append(
                        {"role": "user", "content": q, "time": datetime.now().strftime("%I:%M %p")}
                    )
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        timestamp = msg.get("time", "")
        if msg["role"] == "user":
            st.markdown(
                f"<div class='bubble-label bubble-label-right'>You</div><div class='bubble-user'>{msg['content']}</div><div class='timestamp timestamp-right'>{timestamp}</div><div class='clearfix'></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='bubble-label'>Assistant</div><div class='bubble-bot'>{msg['content']}</div><div class='timestamp'>{timestamp}</div><div class='clearfix'></div>",
                unsafe_allow_html=True,
            )

    question = st.text_area(QUESTION_PLACEHOLDER, height=140, key="prompt_input")

    if st.button("Send", width="stretch"):
        if not question.strip():
            st.warning("Please enter a question before sending.")
            return

        st.session_state.messages.append(
            {"role": "user", "content": question, "time": datetime.now().strftime("%I:%M %p")}
        )

        source_contexts: list[tuple[str, str]] = []
        previews: list[str] = []
        page_filters = {
            int(p.strip()) for p in page_filter_input.split(",") if p.strip().isdigit()
        }

        for source in sidebar_sources:
            label = source["label"]
            index = rag_indexes[label]
            candidates = index.search(question, top_k=top_k * 2)
            if page_filters:
                candidates = [
                    c for c in candidates if c.get("metadata", {}).get("page") in page_filters
                ]

            terms = [t.lower() for t in question.split()]

            def lex_score(chunk: dict) -> int:
                text = chunk.get("chunk", "").lower()
                return sum(text.count(term) for term in terms)

            candidates.sort(key=lambda c: (lex_score(c), c.get("score", 0)), reverse=True)
            reference_chunks = candidates[:top_k]
            if reference_chunks:
                context_text = _format_reference_sections(reference_chunks, source["kind"], label)
                source_contexts.append((label, context_text))
                previews.append(f"{label}:\n{context_text}")

        if source_contexts:
            with st.expander("Validation snippets from all sources", expanded=True):
                st.text_area(
                    "Top matches",
                    value="\n\n".join(previews),
                    height=260,
                    max_chars=None,
                    key="doc_view",
                )
        else:
            st.info("No semantically similar documents were found in the archive.")

        messages = build_generation_messages(source_contexts, question)

        with st.spinner("Sending request to Azure OpenAI chat deployment..."):
            response = openai.chat.completions.create(
                model=deployment_id,
                messages=messages,
                temperature=1.0,
            )
        reply = response.choices[0].message.content
        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "time": datetime.now().strftime("%I:%M %p")}
        )
        st.rerun()


if __name__ == "__main__":
    main()
