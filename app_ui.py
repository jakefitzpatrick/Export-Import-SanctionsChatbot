from __future__ import annotations
import streamlit as st

def get_css() -> str:
    return '''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #f5f7fb; }

    /* Kill ALL outer scrolling */
    html, body { overflow: hidden; height: 100%; }
    .stApp { overflow: hidden; height: 100vh; }
    .stApp > div { overflow: hidden; height: 100vh; }
    section.main { overflow: hidden; height: 100vh; }
    section.main > div.block-container {
        height: 100vh;
        overflow: hidden;
        padding: 0;
        display: flex;
        flex-direction: column;
    }

    /* Fixed top context bar */
    div[data-testid="stVerticalBlock"]:has(> div[data-context="true"]) {
        flex-shrink: 0;
        background: #ffffff;
        border-bottom: 1px solid #e2e8f0;
        padding: 14px 28px 12px;
        box-shadow: 0 2px 8px rgba(15,31,56,0.06);
        z-index: 10;
    }

    /* Scrollable chat middle */
    div[data-testid="stVerticalBlock"]:has(> div[data-chat="true"]) {
        flex: 1;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 16px 28px;
        background: #f5f7fb;
        min-height: 0;
    }

    /* Fixed bottom composer */
    div[data-testid="stVerticalBlock"]:has(> div[data-composer="true"]) {
        flex-shrink: 0;
        background: #ffffff;
        border-top: 1px solid #e2e8f0;
        padding: 10px 28px 14px;
        box-shadow: 0 -4px 12px rgba(15,31,56,0.05);
        z-index: 10;
    }

    [data-testid="stSidebar"] { background: #0f1f38; }
    [data-testid="stSidebar"] * { color: #e2e8f0; }
    [data-testid="stSidebar"] img { display: block; filter: brightness(0) invert(1); mix-blend-mode: normal; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); margin: 10px 0; }
    [data-testid="stSidebar"] h1 { font-size: 22px; font-weight: 700; color: #ffffff; }
    [data-testid="stSidebar"] .sidebar-subtitle { font-size: 12px; color: #94a3b8; line-height: 1.4; }
    [data-testid="stSidebar"] .sidebar-disclaimer { background: rgba(240,165,0,0.12); border-left: 3px solid #f0a500; border-radius: 6px; padding: 8px 12px; font-size: 11px; color: #fde68a; line-height: 1.5; margin-top: 8px; }
    [data-testid="stSidebar"] .sidebar-section-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; }
    [data-testid="stSidebar"] .stButton > button { background-color: rgba(255,255,255,0.08); color: #e2e8f0; border: 1px solid rgba(255,255,255,0.15); border-radius: 10px; font-weight: 500; width: 100%; }
    [data-testid="stSidebar"] .stButton > button:hover { background-color: rgba(255,255,255,0.15); }

    .stButton > button { background-color: #1a3a5c; color: white; border-radius: 10px; font-weight: 500; border: none; padding: 10px; }
    .stButton > button:hover { background-color: #2a5298; }

    .bubble-user { background-color: #1a3a5c; color: white; padding: 12px 18px; border-radius: 18px 18px 4px 18px; margin: 4px 0; max-width: 70%; float: right; clear: both; font-size: 14px; line-height: 1.5; }
    .bubble-bot { background-color: #ffffff; color: #1a1a2e; padding: 12px 18px; border-radius: 18px 18px 18px 4px; margin: 4px 0; max-width: 70%; float: left; clear: both; font-size: 14px; line-height: 1.5; border: 1px solid #e2e8f0; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
    .bubble-label { font-size: 10px; color: #94a3b8; margin-bottom: 2px; clear: both; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
    .bubble-label-right { text-align: right; }
    .timestamp { font-size: 10px; color: #cbd5e1; margin-top: 3px; clear: both; }
    .timestamp-right { text-align: right; }
    .clearfix { clear: both; }
    .risk-pills { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 4px; }
    .risk-pill { border: 1px solid #e2e8f0; border-left: 4px solid #e2e8f0; padding: 8px 12px; border-radius: 12px; background: #f9fafc; font-size: 12px; font-weight: 500; }
    .risk-pill strong { display: block; font-size: 12px; color: #0f1f38; }
    .analysis-meta { font-size: 12px; color: #94a3b8; margin-bottom: 6px; }
    .empty-chat { color: #94a3b8; font-size: 13px; text-align: center; padding: 40px 0; }
    div[data-anchor="chat-end"] { display: none; }
    img { mix-blend-mode: multiply; }
    </style>
    '''

def render_sidebar(on_clear) -> None:
    with st.sidebar:
        try:
            st.image("logo.png", width=140)
        except Exception:
            pass
        st.markdown("<h1>ImportInsight AI</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sidebar-subtitle'>Trade and tariff intelligence powered by your HTS SQLite database.</p>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-disclaimer'><b>Disclaimer:</b> This tool is for informational purposes only and does not constitute legal advice. Always consult a qualified trade compliance professional.</div>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<p class='sidebar-section-label'>Settings</p>", unsafe_allow_html=True)
        st.caption("Natural language inputs are translated to SQL, executed locally against a read-only SQLite copy of the HTS data.")
        st.caption("Responses are deterministic: the SQL output is re-run each time against the local database.")
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("Clear Chat", key="sidebar_clear", use_container_width=True):
            on_clear()
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<p class='sidebar-section-label'>About</p>", unsafe_allow_html=True)
        st.caption("ImportInsight AI translates your prompt into SQL and returns the actual HTS rows with governance risk context.")
