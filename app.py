import os
from pathlib import Path
from dotenv import load_dotenv
import openai
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from rag import DEFAULT_EMBEDDING_MODEL, RagIndex

load_dotenv()

RAG_ARCHIVE = Path("SDN_ENHANCED.ZIP")
DEFAULT_TOP_K = 3

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
        return "#c0392b", "High"
    elif score >= 45:
        return "#d35400", "Medium"
    else:
        return "#1e8449", "Low"

def _format_reference_sections(chunks):
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[{i}] {chunk}")
        lines.append("")
    return "\n".join(lines)

def main() -> None:
    st.set_page_config(page_title="ImportInsight AI", layout="wide")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] {
        background-color: #0f1f38;
        padding-top: 0 !important;
    }
    [data-testid="stSidebar"] * { color: #e8edf5 !important; }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #1e3a5f !important;
        border: 1px solid #2d5a8e !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stSelectbox input {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] hr { border-color: #1e3a5f !important; }
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
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <h1 style='color:#0f1f38; font-weight:700; letter-spacing:-0.5px;'>ImportInsight AI</h1>
        <p style='color:#64748b; font-size:15px; margin-top:-10px;'>Trade and sanctions intelligence powered by OFAC SDN data.</p>
        <hr style='border: 1px solid #e2e8f0; margin-top:16px;'>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
        <div style='background-color:#fef9ec; border-left: 4px solid #f0a500; padding: 10px 16px; border-radius: 6px; margin-bottom: 20px;'>
            <span style='color:#7d5a00; font-size:13px;'>
                <b>Disclaimer:</b> This tool is for informational purposes only and does not constitute legal advice. Always consult a qualified trade compliance professional.
            </span>
        </div>
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
        st.image("logo.png", width=150)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### Settings")
        top_k = st.slider("Chunks to consider", 1, 5, DEFAULT_TOP_K)
        st.caption("Searches SDN entities for semantic matches.")
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
            number={'font': {'size': 28, 'color': color}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#a0aec0", 'tickfont': {'size': 9, 'color': '#a0aec0'}},
                'bar': {'color': color, 'thickness': 0.25},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 45], 'color': '#1e8449'},
                    {'range': [45, 75], 'color': '#d35400'},
                    {'range': [75, 100], 'color': '#c0392b'},
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.8,
                    'value': score
                }
            },
            title={'text': f"<b>{level} Risk</b><br><span style='font-size:11px;color:#a0aec0'>{selected_country} · {year}</span>", 'font': {'size': 13, 'color': 'white'}}
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=60, b=10),
            height=200,
            font={'color': 'white'}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### About")
        st.caption("ImportInsight AI helps businesses navigate trade and sanctions regulations using OFAC data.")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Starter pills
    if len(st.session_state.messages) == 0:
        starters = [
            "Can I export software to Russia?",
            "Is Huawei on the entity list?",
            "What tariffs apply to Chinese electronics?",
            "How do I screen a foreign supplier?",
            "What goods are banned from North Korea?",
            "Are there restrictions on trading with Iran?",
        ]
        st.markdown("<p style='color:#cbd5e1; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:6px;'>Suggested questions</p>", unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate(starters):
            with cols[i % 2]:
                if st.button(q, key=f"s_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q, "time": datetime.now().strftime("%I:%M %p")})
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    # Display chat bubbles
    for msg in st.session_state.messages:
        timestamp = msg.get("time", "")
        if msg["role"] == "user":
            st.markdown(f"<div class='bubble-label bubble-label-right'>You</div><div class='bubble-user'>{msg['content']}</div><div class='timestamp timestamp-right'>{timestamp}</div><div class='clearfix'></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bubble-label'>Assistant</div><div class='bubble-bot'>{msg['content']}</div><div class='timestamp'>{timestamp}</div><div class='clearfix'></div>", unsafe_allow_html=True)

    # Input
    question = st.text_area("Ask about sanctions, trade restrictions, or SDN entities:", height=100, key="prompt_input")

    if st.button("Send", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question before sending.")
            return

        now = datetime.now().strftime("%I:%M %p")
        st.session_state.messages.append({"role": "user", "content": question, "time": now})

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
        st.session_state.messages.append({"role": "assistant", "content": reply, "time": datetime.now().strftime("%I:%M %p")})
        st.rerun()

if __name__ == "__main__":
    main()