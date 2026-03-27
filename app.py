import os
from pathlib import Path
from dotenv import load_dotenv
import openai
import streamlit as st
from rag import DEFAULT_EMBEDDING_MODEL, RagIndex

load_dotenv()

RAG_ARCHIVE = Path("SDN_ENHANCED.ZIP")
DEFAULT_TOP_K = 3

# Governance risk data
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

def get_risk_color(score):
    if score >= 75:
        return "#e53935", "High"
    elif score >= 45:
        return "#fb8c00", "Medium"
    else:
        return "#43a047", "Low"

def _format_reference_sections(chunks):
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[{i}] {chunk}")
        lines.append("")
    return "\n".join(lines)

def main() -> None:
    st.set_page_config(page_title="Sanctions RAG Assistant", layout="wide")

    st.markdown("""
    <style>
    body { background-color: #f4f6f9; }
    .bubble-user {
        background-color: #1a3a5c;
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 75%;
        float: right;
        clear: both;
        font-size: 15px;
    }
    .bubble-bot {
        background-color: #ffffff;
        color: #1a1a1a;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 75%;
        float: left;
        clear: both;
        font-size: 15px;
        border: 1px solid #dce3ec;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .bubble-label { font-size: 11px; color: #888; margin-bottom: 2px; clear: both; }
    .bubble-label-right { text-align: right; }
    .clearfix { clear: both; }
    .risk-card {
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        margin-top: 10px;
    }
    .risk-score-number {
        font-size: 52px;
        font-weight: bold;
        line-height: 1;
    }
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        background: rgba(255,255,255,0.25);
        font-size: 14px;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <h1 style='color:#1a3a5c;'>Sanctions RAG Assistant</h1>
        <p style='color:#555;'>Uses the OFAC SDN ENHANCED XML archive as the knowledge source.</p>
        <hr style='border: 1px solid #dce3ec;'>
    """, unsafe_allow_html=True)

    # Azure OpenAI setup
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
        st.error("Please set AZURE_OPENAI_DEPLOYMENT_ID before running the app.")
        return

    if not RAG_ARCHIVE.exists():
        st.warning("Please drop SDN_ENHANCED.ZIP into the app root so the RAG index can be built.")
        return

    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_ID", DEFAULT_EMBEDDING_MODEL)

    try:
        rag_index = RagIndex(RAG_ARCHIVE, embedding_deployment)
    except Exception as exc:
        st.error(f"Failed to prepare RAG index: {exc}")
        return

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        top_k = st.slider("Chunks to consider", 1, 5, DEFAULT_TOP_K)
        st.caption("Searches SDN entities for semantic matches.")
        st.markdown("---")

        # Governance Risk Panel
        st.markdown("### 🌍 Governance Risk")
        selected_country = st.selectbox("Select country", list(COUNTRY_RISK.keys()))
        data = COUNTRY_RISK[selected_country]
        score = data["score"]
        year = data["year"]
        color, level = get_risk_color(score)

        st.markdown(f"""
        <div class='risk-card' style='background-color:{color};'>
            <div style='font-size:13px; opacity:0.85;'>Risk Score</div>
            <div class='risk-score-number'>{score}</div>
            <div class='risk-badge'>{level} Risk</div>
            <div style='font-size:12px; margin-top:10px; opacity:0.8;'>
                {selected_country} · {year}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📋 About")
        st.caption("This assistant helps businesses understand trade and sanctions regulations using OFAC data.")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='bubble-label bubble-label-right'>You</div><div class='bubble-user'>{msg['content']}</div><div class='clearfix'></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bubble-label'>Assistant</div><div class='bubble-bot'>{msg['content']}</div><div class='clearfix'></div>", unsafe_allow_html=True)

    # Input
    question = st.text_area("Ask about sanctions, trade restrictions, or SDN entities:", height=100, key="prompt_input")

    if st.button("Send", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question before sending.")
            return

        st.session_state.messages.append({"role": "user", "content": question})

        reference_chunks = rag_index.search(question, top_k=top_k)
        if reference_chunks:
            context_text = _format_reference_sections(reference_chunks)
        else:
            context_text = ""
            st.info("No semantically similar documents were found.")

        messages = [{"role": "system", "content": "You answer questions using the OFAC SDN Enhanced data; cite relevant retrieved evidence when available."}]
        if context_text:
            messages.append({"role": "system", "content": "Reference data:\n" + context_text})
        messages.append({"role": "user", "content": question})

        with st.spinner("Thinking..."):
            response = openai.chat.completions.create(
                model=deployment_id,
                messages=messages,
                temperature=1.0,
            )
        reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

if __name__ == "__main__":
    main()