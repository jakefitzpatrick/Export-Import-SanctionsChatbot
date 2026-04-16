from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


def get_css() -> str:
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    :root {
        --navy:#0B2A4A;
        --muted:#94A3AF;
        --border:#E5E7EB;
        --card:#FFFFFF;
        --accent:#1A3A5C;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background:#ffffff;
        color:#0B2239;
    }
    .stApp, .stApp > div, section.main { min-height: 100vh; background:#ffffff; }
    section.main > div.block-container {
        display:flex;
        flex-direction:column;
        height:100vh;
        padding:0;
    }
    div[data-testid="stVerticalBlock"]:has(> div[data-context="true"]) {
        flex-shrink:0;
        padding:14px 28px 12px;
        border-bottom:1px solid var(--border);
        background:#fff;
        z-index:20;
    }
    div[data-testid="stVerticalBlock"]:has(> div[data-chat="true"]) {
        flex:1;
        overflow-y:auto;
        padding:0 28px 16px;
        background:#f7f8fb;
    }
    div[data-testid="stVerticalBlock"]:has(> div[data-composer="true"]) {
        flex-shrink:0;
        padding:12px 28px 18px;
        border-top:1px solid var(--border);
        background:#fff;
        box-shadow:0 -4px 16px rgba(11,34,57,0.05);
    }
    [data-testid="stSidebar"] { background: var(--navy); color:#e7edf5; }
    [data-testid="stSidebar"] img { filter:brightness(0) invert(1); margin-bottom:12px; }
    .sidebar-mono { font-size:9.5px; text-transform:uppercase; letter-spacing:0.08em; color:rgba(255,255,255,0.35); }
    .sidebar-tag { display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px; font-size:11px; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.18); color:#F8FAFC; }
    .bubble-user, .bubble-bot {
        transition:transform 0.15s ease, box-shadow 0.15s ease;
        border-radius:18px;
        padding:12px 18px;
        max-width:72%;
        line-height:1.5;
        font-size:14px;
    }
    .bubble-user { background:#1a3a5c; color:#fff; margin:4px 0 4px auto; box-shadow:0 2px 6px rgba(26,58,92,0.25); }
    .bubble-bot { background:#fff; color:#0f172a; border:1px solid #e2e8f0; margin:4px auto 4px 0; box-shadow:0 2px 8px rgba(0,0,0,0.05); }
    .bubble-bot-wrap:hover .bubble-bot, .bubble-user:hover {
        transform:translateY(-2px);
        box-shadow:0 8px 20px rgba(15,31,56,0.12);
    }
    .bubble-label { font-size:10px; letter-spacing:0.08em; text-transform:uppercase; color:var(--muted); margin-bottom:2px; }
    .bubble-label-right { text-align:right; }
    .timestamp, .timestamp-right { font-size:10px; color:#cbd5e1; margin-top:4px; }
    .timestamp-right { text-align:right; }
    .analysis-meta { font-size:12px; color:#94a3b8; margin-bottom:6px; }
    .analysis-headline { font-weight:600; margin-bottom:6px; color:#0B2A4A; }
    .empty-chat-card { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:14px; height:340px; color:#6B7280; }
    div[data-anchor="chat-end"] { display:none; }
    .risk-gauge-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:12px 0; }
    .risk-gauge-card { background:#fff; border:1px solid var(--border); border-radius:12px; padding:14px; box-shadow:0 4px 14px rgba(15,31,56,0.08); }
    .risk-gauge-track { position:relative; height:8px; border-radius:999px; overflow:hidden; background:#f3f4f6; margin:8px 0 4px; }
    .risk-gauge-track span { position:absolute; inset:0; }
    .risk-gauge-needle { position:absolute; top:-4px; width:3px; height:16px; background:#0B2A4A; border-radius:999px; }
    </style>
    """

def render_sidebar(on_clear) -> None:
    """Branding, disclaimer, session history, and the sidebar actions."""
    with st.sidebar:
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] .history-item {
                padding:6px 8px;
                border-radius:6px;
                background:rgba(255,255,255,0.06);
                color:rgba(255,255,255,0.6);
                font-size:12px;
                margin-bottom:4px;
            }
            [data-testid="stSidebar"] .history-item span {
                color:#4F8FB8;
                margin-right:6px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        try:
            st.image("logo_clean.png", width=130)
        except Exception:
            st.image("logo.png", width=130)
        st.markdown(
            "<p class='sidebar-mono' style='margin-top:-8px;'>ImportInsight AI</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='sidebar-tag'>Trade data · HTS SQLite</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='margin:12px 0;background:rgba(240,165,0,0.15);"
            "border-left:3px solid #f0a500;border-radius:6px;padding:8px 12px;"
            "font-size:11px;color:#fde68a;'>"
            "<b>Disclaimer:</b> Informational only — not legal advice.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<p class='sidebar-mono' style='margin-top:18px;'>Recent sessions</p>", unsafe_allow_html=True)
        session_history = st.session_state.get("session_history", [])
        if session_history:
            for entry in reversed(session_history[-4:]):
                st.markdown(
                    f"<div class='history-item'><span>●</span>{entry}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div style='font-size:11px;color:rgba(255,255,255,0.35);padding:4px 8px;'>No sessions yet</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<p class='sidebar-mono'>About</p>", unsafe_allow_html=True)
        st.caption("Natural-language prompts are translated to SQL and executed locally against your read-only HTS SQLite database.")
        st.caption("Results are deterministic — every answer reruns against the local data.")
        if st.button("Clear Chat", key="sidebar_clear", use_container_width=True):
            on_clear()



def render_context_panel(
    *,
    country_options: list[str],
    display_options: list[str],
    code_map: dict[str, str],
    max_country: int,
    max_products: int,
    country_trimmed: bool,
    product_trimmed: bool,
    analysis_running: bool,
    on_clear_chat: Callable[[], None],
    logger,
) -> tuple[list[str], bool]:
    context_bar = st.container()
    with context_bar:
        st.markdown('<div data-context="true"></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div style='display:flex;align-items:center;justify-content:space-between;padding:6px 0 14px;border-bottom:1px solid #E5E7EB;margin-bottom:14px;'>
              <div style='display:flex;align-items:center;gap:10px;'>
                <span style='font-family:serif;font-size:18px;font-weight:700;color:#0B2A4A;letter-spacing:-0.3px;'>HTS Analysis</span>
                <span style='font-size:11px;padding:3px 10px;border-radius:999px;background:#F0FDF4;border:0.5px solid #86EFAC;color:#15803D;display:inline-flex;align-items:center;gap:6px;'>
                  <span style='position:relative;width:8px;height:8px;display:inline-block;'>
                    <span style='position:absolute;inset:0;border-radius:50%;background:#22C55E;opacity:0.4;animation:ping 1.5s cubic-bezier(0,0,0.2,1) infinite;'></span>
                    <span style='position:absolute;inset:1px;border-radius:50%;background:#16A34A;display:inline-block;'></span>
                  </span>
                  Connected
                </span>
              </div>
            </div>
            <style>
            @keyframes ping { 0% { transform: scale(1); opacity: 0.4; } 75%,100% { transform: scale(2); opacity: 0; } }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Add up to three countries and three HTS codes, then run an aligned analysis or ask a direct question.")
        sel_cols = st.columns(2, gap="large")
        with sel_cols[0]:
            st.markdown("<p style='font-size:12px;font-weight:600;color:#0B2A4A;margin-bottom:4px;'>Countries</p>", unsafe_allow_html=True)
            st.multiselect(
                "Countries",
                options=country_options,
                key="selected_countries",
                max_selections=max_country,
                label_visibility="collapsed",
                help="Choose up to three countries to compare governance risk.",
            )
        with sel_cols[1]:
            st.markdown("<p style='font-size:12px;font-weight:600;color:#0B2A4A;margin-bottom:4px;'>HTS Products</p>", unsafe_allow_html=True)
            if display_options:
                st.multiselect(
                    "HTS Products",
                    options=display_options,
                    key="selected_products_display",
                    max_selections=max_products,
                    label_visibility="collapsed",
                    help="Products are loaded from the local HTS SQLite database.",
                )
            else:
                st.error("No HTS products found — rebuild the SQLite database.")
        selected_countries = st.session_state.get("selected_countries", [])
        selected_products_display = st.session_state.get("selected_products_display", [])
        st.caption(
            f"{len(selected_countries)} / {max_country} countries · {len(selected_products_display)} / {max_products} products"
        )
        if country_trimmed:
            st.warning(f"Country selection limited to {max_country}. Extra choices were dropped.")
            logger.info("Trimmed country selection to limit", extra={"limit": max_country})
        if product_trimmed:
            st.warning(f"Product selection limited to {max_products}. Extra choices were dropped.")
            logger.info("Trimmed product selection to limit", extra={"limit": max_products})
        current_product_labels = st.session_state.get("selected_products_display", [])
        valid_product_labels = [label for label in current_product_labels if label in code_map]
        if len(valid_product_labels) != len(current_product_labels):
            st.warning("Some selected products are unavailable in the current HTS list.")
        selected_products = [code_map[label] for label in valid_product_labels]
        st.session_state["selected_product_codes"] = selected_products

        action_cols = st.columns([1, 1], gap="large")
        analyse_disabled = analysis_running or not selected_countries or not selected_products
        with action_cols[0]:
            analyse_clicked = st.button(
                "Analyze",
                disabled=analyse_disabled,
                use_container_width=True,
            )
        with action_cols[1]:
            if st.button("Clear Inputs", key="context_clear", use_container_width=True):
                on_clear_chat()
        if analysis_running:
            st.caption("Running analysis…")
    return selected_products, analyse_clicked


def render_chat_composer(placeholder: str) -> str | None:
    composer = st.container()
    with composer:
        st.markdown('<div data-composer="true"></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <style>
            [data-testid="stChatInputContainer"] {
                background:#FFFFFF !important;
                border:2px solid #CBD5E1 !important;
                border-radius:32px !important;
                padding:6px 10px 6px 20px !important;
                outline:4px solid rgba(11,42,74,0.08) !important;
            }
            [data-testid="stChatInputContainer"]:focus-within {
                border-color:#0B2A4A !important;
                outline:4px solid rgba(11,42,74,0.12) !important;
            }
            [data-testid="stChatInputContainer"] textarea {
                color:#1F2937 !important;
                font-size:14px !important;
                background:transparent !important;
            }
            [data-testid="stChatInputContainer"] textarea::placeholder {
                color:#9CA3AF !important;
                opacity:1 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return st.chat_input(placeholder, key="chat_input")


def _render_risk_gauges(risk_snapshot: list[dict]) -> None:
    if not risk_snapshot:
        return
    def _needle(score: float) -> float:
        score = max(0.0, min(float(score), 100.0))
        return score

    def _badge(level: str) -> tuple[str, str]:
        if not level:
            return "#E5E7EB", "#1F2937"
        level = level.lower()
        if level == "high":
            return "#FCEBEB", "#7A1F1F"
        if level in {"medium", "moderate"}:
            return "#FAEEDA", "#633806"
        return "#EAF3DE", "#27500A"

    cards: list[str] = []
    for snap in risk_snapshot:
        level = snap.get("level", "Unknown")
        badge_bg, badge_color = _badge(level)
        score = round(float(snap.get("score", 0.0)), 1)
        cards.append(
            f"""
            <div class='risk-gauge-card'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <span style='font-size:13px;font-weight:600;color:#0B2A4A;'>{snap.get('country','—')}</span>
                    <span style='font-size:10.5px;font-weight:500;padding:2px 9px;border-radius:999px;background:{badge_bg};color:{badge_color};'>{level}</span>
                </div>
                <div class='risk-gauge-track'>
                    <span style='left:0;width:33%;background:#1D9E75;'></span>
                    <span style='left:33%;width:34%;background:#BA7517;'></span>
                    <span style='left:67%;width:33%;background:#D85A30;'></span>
                    <div class='risk-gauge-needle' style='left:{_needle(score)}%;'></div>
                </div>
                <div style='display:flex;justify-content:space-between;font-size:9.5px;color:#9CA3AF;margin-top:4px;'>
                    <span>Low</span><span>Medium</span><span>High</span>
                </div>
                <div style='margin-top:10px;'>
                    <span style='font-family:monospace;font-size:26px;font-weight:500;color:#0B2A4A;'>{score}</span>
                    <span style='font-size:12px;color:#9CA3AF;margin-left:3px;'>/ 100</span>
                </div>
            </div>
            """
        )
    st.markdown(
        f"<div class='risk-gauge-grid'>{''.join(cards)}</div>",
        unsafe_allow_html=True,
    )


def render_chat_feed(messages: list[dict], latest_result: dict | None):
    chat_feed = st.container()
    with chat_feed:
        st.markdown('<div data-chat="true"></div>', unsafe_allow_html=True)
        if len(messages) == 0:
            st.markdown(
                """
                <div class='empty-chat-card'>
                  <div style='width:56px;height:56px;border-radius:14px;border:2px solid #D1D5DB;display:flex;align-items:center;justify-content:center;'>
                    <svg width=\"24\" height=\"24\" viewBox=\"0 0 24 24\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\">
                      <path d=\"M21 15C21 15.5304 20.7893 16.0391 20.4142 16.4142C20.0391 16.7893 19.5304 17 19 17H7L3 21V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H19C19.5304 3 20.0391 3.21071 20.4142 3.58579C20.7893 3.96086 21 4.46957 21 5V15Z\" stroke=\"#9CA3AF\" stroke-width=\"1.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>
                    </svg>
                  </div>
                  <div style='text-align:center;'>
                    <p style='font-size:15px;font-weight:600;color:#374151;margin:0 0 6px;'>Select countries and HTS products, then click <span style="color:#0B2A4A;">Analyze</span> to begin.</p>
                    <p style='font-size:13px;color:#9CA3AF;margin:0;'>Or type a question in the composer below.</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        for msg in messages:
            timestamp = msg.get('time', '')
            if msg.get('role') == 'user':
                st.markdown(
                    f"<div class='bubble-label bubble-label-right'>You</div><div class='bubble-user'>{msg['content']}</div><div class='timestamp timestamp-right'>{timestamp}</div><div class='clearfix'></div>",
                    unsafe_allow_html=True,
                )
                continue

            if msg.get('type') == 'analysis':
                st.markdown("<div class='bubble-label'>Assistant</div>", unsafe_allow_html=True)
                selections = msg.get('selections', {})
                selection_text = []
                if selections.get('countries'):
                    selection_text.append('Countries: ' + ', '.join(selections['countries']))
                if selections.get('products'):
                    selection_text.append('Products: ' + ', '.join(selections['products']))
                if selection_text:
                    st.markdown(
                        f"<div class='analysis-meta'>{' · '.join(selection_text)}</div>",
                        unsafe_allow_html=True,
                    )
                fig_payload = msg.get('plotly_fig')
                if fig_payload:
                    fig = go.Figure(fig_payload)
                    st.plotly_chart(fig, use_container_width=True)
                body_html = msg['content']
                headline = msg.get('headline')
                if headline:
                    body_html = f"<p class='analysis-headline'><strong>{headline}</strong></p>{body_html}"
                st.markdown(
                    f"<div class='bubble-bot-wrap'><div class='bubble-bot'>{body_html}</div></div><div class='timestamp'>{timestamp}</div><div class='clearfix'></div>",
                    unsafe_allow_html=True,
                )
                non_ad_summary_text = msg.get('non_ad_summary_text')
                if non_ad_summary_text:
                    st.info(non_ad_summary_text)
                duty_exclusion_message = msg.get('duty_exclusion_message')
                duty_exclusions = msg.get('duty_exclusions') or []
                if duty_exclusion_message:
                    st.warning(duty_exclusion_message)
                elif duty_exclusions:
                    listed = ', '.join(
                        f"{item.get('hts_code')} ({item.get('general_duty_rate_text')})"
                        for item in duty_exclusions[:3]
                    )
                    if len(duty_exclusions) > 3:
                        listed += f", +{len(duty_exclusions) - 3} more"
                    st.warning(f"Skipped non-percentage duty rates: {listed}")
                ch99_summary = msg.get('ch99_summary') or {}
                if ch99_summary.get('n_adjusted'):
                    n_total = ch99_summary.get('n_total', ch99_summary['n_adjusted'])
                    programs = ch99_summary.get('programs') or []
                    program_text = f" ({', '.join(programs)})" if programs else ''
                    st.info(f"Chapter 99 adjustments applied to {ch99_summary['n_adjusted']} of {n_total} plotted points{program_text}.")
                _render_risk_gauges(msg.get('risk_snapshot') or [])
                chart_data = msg.get('chart_data')
                if chart_data:
                    chart_df = pd.DataFrame(chart_data, columns=msg.get('chart_columns'))
                    st.dataframe(chart_df, use_container_width=True)
                continue

            st.markdown(
                f"<div class='bubble-label'>Assistant</div><div class='bubble-bot-wrap'><div class='bubble-bot'>{msg['content']}</div></div><div class='timestamp'>{timestamp}</div><div class='clearfix'></div>",
                unsafe_allow_html=True,
            )

        analysis_stream_placeholder = st.empty()
        if latest_result:
            st.markdown('---')
            st.markdown('### Last SQL Result')
            st.code(latest_result['sql'], language='sql')
            st.caption(
                f"Returned {latest_result['row_count']} row(s); showing {latest_result['rows_displayed']} row(s) below."
            )
            if latest_result['records']:
                st.dataframe(
                    pd.DataFrame(latest_result['records'], columns=latest_result['columns']),
                    use_container_width=True,
                )
            else:
                st.info('The last query returned no rows.')
        st.markdown('<div data-anchor="chat-end" id="chat-end"></div>', unsafe_allow_html=True)
        if analysis_stream_placeholder is None:
            analysis_stream_placeholder = st.empty()
    components.html(
        """
        <script>
        const marker = window.parent.document.querySelector('div[data-anchor="chat-end"]');
        if (marker) {
            const chatBlock = marker.closest('div[data-testid="stVerticalBlock"]');
            if (chatBlock) {
                chatBlock.scrollTop = chatBlock.scrollHeight;
            }
        }
        </script>
        """,
        height=0,
    )
    return analysis_stream_placeholder
