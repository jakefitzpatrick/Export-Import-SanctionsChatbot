from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_inline_iframe(html: str, *, height: int = 1) -> None:
    """Embed inline HTML using st.components.v1.html."""
    import streamlit.components.v1 as components
    components.html(html, height=height, scrolling=False)


def get_css() -> str:
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    :root {
        --navy:#0B2A4A;
        --border:#CBD5E1;
        --border-strong:#94A3B8;
        --card:#FFFFFF;
        --surface-muted:#EEF2F7;
        --accent:#0B2A4A;
        --text-primary:#0F172A;
        --text-secondary:#1F2937;
        --text-muted:#374151;
        --chip-selected-bg:#0B2A4A;
        --chip-selected-border:#062340;
        --chip-focus:#1D4ED8;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background:#ffffff;
        color:var(--text-primary);
    }
    .stApp, .stApp > div, section.main { min-height: 100vh; background:#ffffff; }
    section.main > div.block-container {
        display:flex;
        flex-direction:column;
        min-height:100vh;
        padding:0;
    }
    section.main > div.block-container > div[data-testid="stVerticalBlock"] {
        display:flex;
        flex-direction:column;
        flex:0 0 auto;
    }
    section.main > div.block-container > div[data-testid="stVerticalBlock"] > .layout-zone {
        width:100%;
    }
    section.main > div.block-container > div[data-testid="stVerticalBlock"] > .layout-zone--chat {
        flex:1 1 auto;
        min-height:0;
    }
    .layout-zone--inputs {
        order:1;
        flex-shrink:0;
        padding:10px 28px 8px;
        border-bottom:1px solid var(--border);
        background:#fff;
        z-index:20;
    }
    .layout-zone--chat {
        order:2;
        flex:1 1 auto;
        min-height:0;
        padding:8px 28px 8px;
        background:#ffffff;
        display:flex;
        flex-direction:column;
    }
    .layout-zone--composer {
        order:3;
        flex-shrink:0;
        padding:6px 28px 12px;
        border-top:none;
        background:transparent;
        box-shadow:none;
    }
    .layout-zone--chat div[data-chat="true"] {
        flex:1;
        overflow-y:auto;
        background:#FDFDFE;
        border-radius:18px;
        padding:4px 12px 16px;
        box-shadow:0 8px 24px rgba(15,23,42,0.08);
    }
    [data-testid="stSidebar"] { background: var(--navy); color:#E2E8F0; }
    [data-testid="stSidebar"] img { filter:brightness(0) invert(1); margin-bottom:12px; }
    .sidebar-mono { font-size:9.5px; text-transform:uppercase; letter-spacing:0.08em; color:#CBD5F5; }
    .sidebar-tag { display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:999px; font-size:11px; background:#F8FAFC; border:1px solid #CBD2D9; color:#0F2342; font-weight:600; }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stCaption {
        color:#F5F7FF !important;
    }
    .bubble-user, .bubble-bot {
        transition:transform 0.15s ease, box-shadow 0.15s ease;
        border-radius:18px;
        padding:12px 18px;
        max-width:72%;
        line-height:1.5;
        font-size:14px;
    }
    .bubble-user { background:#1a3a5c; color:#fff; margin:4px 0 4px auto; box-shadow:0 2px 6px rgba(26,58,92,0.25); }
    .bubble-bot { background:#fff; color:var(--text-primary); border:1px solid #E2E8F0; margin:4px auto 4px 0; box-shadow:0 6px 18px rgba(15,23,42,0.08); }
    .bubble-bot-wrap:hover .bubble-bot, .bubble-user:hover {
        transform:translateY(-2px);
        box-shadow:0 8px 20px rgba(15,31,56,0.12);
    }
    .bubble-label { font-size:10px; letter-spacing:0.08em; text-transform:uppercase; color:var(--text-secondary); margin-bottom:2px; }
    .bubble-label-right { text-align:right; }
    .timestamp, .timestamp-right { font-size:10px; color:var(--text-muted); margin-top:4px; }
    .timestamp-right { text-align:right; }
    .analysis-meta { font-size:12px; color:#1F2937; margin-bottom:6px; }
    .analysis-headline { font-weight:600; margin-bottom:6px; color:#031B4D; }
    .empty-chat-card { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:14px; height:340px; color:var(--text-secondary); }
    div[data-anchor="chat-end"] { display:none; }
    .risk-gauge-grid { display:flex; justify-content:center; margin:10px 0 18px; }
    .risk-gauge-card { width:260px; background:#fff; border:1px solid var(--border); border-radius:16px; padding:12px 16px; box-shadow:0 8px 24px rgba(15,23,42,0.12); }
    .risk-gauge-track { position:relative; height:6px; border-radius:999px; overflow:hidden; background:#E2E8F0; margin:10px 0 6px; }
    .risk-gauge-track span { position:absolute; inset:0; }
    .risk-gauge-needle { position:absolute; top:-3px; width:3px; height:12px; background:#0B2A4A; border-radius:999px; }
    .stMarkdown, .stMarkdown p, .stText, .stCaption, .stRadio label, .stCheckbox label, .stSelectbox label, .stMultiSelect label {
        color:var(--text-primary) !important;
    }
    .stCaption, .stMarkdown caption {
        color:#1E293B !important;
    }
    .stMultiSelect div[data-baseweb="tag"] {
        background:var(--chip-selected-bg) !important;
        color:#FFFFFF !important;
        border:1px solid var(--chip-selected-border) !important;
        border-radius:999px !important;
        font-weight:600;
        letter-spacing:0.01em;
    }
    .stMultiSelect div[data-baseweb="tag"]:focus-within {
        box-shadow:0 0 0 2px var(--chip-focus) !important;
    }
    .stMultiSelect div[data-baseweb="tag"] svg {
        fill:#FFFFFF !important;
    }
    .stMultiSelect [data-baseweb="menu"] li[aria-selected="true"] {
        background:rgba(11,42,74,0.18) !important;
        color:#0B2A4A !important;
    }
    .stMultiSelect [data-baseweb="menu"] li:hover {
        background:rgba(11,42,74,0.12) !important;
        color:#0B2A4A !important;
    }
    .stRadio div[role="radiogroup"] label {
        color:var(--text-primary) !important;
    }
    .stRadio div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
        color:#B42318 !important;
        font-weight:600;
    }
    .stSelectbox [data-baseweb="select"] > div {
        color:var(--text-primary);
    }
    .stButton button, .stDownloadButton button {
        color:var(--text-primary);
        border:1px solid var(--border-strong);
    }
    .tariff-ctx table {
        border:1px solid #0F172A;
        border-radius:10px;
        overflow:hidden;
        box-shadow:0 6px 18px rgba(15,23,42,0.18);
    }
    .tariff-ctx th {
        background:#E2E8F0;
        color:#050914;
        border-bottom:2px solid #0F172A;
    }
    .tariff-ctx td {
        color:#050914;
        border-bottom:1px solid #94A3B8;
    }
    .tariff-ctx tbody tr:nth-child(even) {
        background:#F4F6FB;
    }
    div[data-testid="stDataFrame"] {
        border-radius:12px;
        border:1px solid #0F172A;
        box-shadow:0 12px 32px rgba(15,23,42,0.16);
        padding:8px;
        background:#fff;
    }
    div[data-testid="stDataFrame"] table {
        color:#050914;
    }
    div[data-testid="stDataFrame"] thead th {
        background:#E3E8F2;
        color:#050914;
        border-bottom:2px solid #0F172A;
        text-transform:uppercase;
        font-size:11px;
        letter-spacing:0.04em;
    }
    div[data-testid="stDataFrame"] tbody td {
        border-bottom:1px solid #94A3B8;
    }
    div[data-testid="stDataFrame"] tbody tr:nth-child(even) {
        background:#F5F6FA;
    }
    div[data-testid="stDataFrame"] tbody tr:hover td {
        background:#E8ECF8;
    }
    </style>
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
            [data-testid="stSidebar"] button {
                background:rgba(255,255,255,0.08);
                color:#F9FAFB;
                border:1px solid rgba(255,255,255,0.25);
                border-radius:999px;
                font-weight:600;
                letter-spacing:0.02em;
                box-shadow:0 4px 12px rgba(0,0,0,0.25);
            }
            [data-testid="stSidebar"] button:hover {
                background:rgba(255,255,255,0.18);
            }
    [data-testid="stSidebar"] button:focus-visible {
        outline:3px solid rgba(255,255,255,0.4);
    }
    [data-testid="collapsedControl"] button {
        background:rgba(15,35,66,0.92) !important;
        border:1px solid rgba(255,255,255,0.25) !important;
        color:#F8FAFC !important;
        box-shadow:0 6px 18px rgba(0,0,0,0.35) !important;
    }
    [data-testid="collapsedControl"] button:hover {
        background:rgba(11,42,74,1) !important;
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
            "<div style='margin:12px 0;background:#FFF7D6;"
            "border-left:3px solid #EAAA08;border-radius:6px;padding:8px 12px;"
            "font-size:11px;color:#1B1F32;'>"
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
                "<div style='font-size:11px;color:#0F2342;font-weight:600;padding:4px 12px;border-radius:999px;background:#F2F4F7;border:1px solid #CBD2D9;display:inline-flex;align-items:center;'>No sessions yet</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<p class='sidebar-mono'>About</p>", unsafe_allow_html=True)
        st.caption("Natural-language prompts are translated to SQL and executed locally against your read-only HTS SQLite database.")
        st.caption("Results are deterministic — every answer reruns against the local data.")
        if st.button("Clear Chat", key="sidebar_clear", width="stretch"):
            on_clear()



def render_context_panel(
    *,
    country_options: list[str],
    product_options: list[str],
    code_map: dict[str, str],
    max_country: int,
    max_products: int,
    country_trimmed: bool,
    specific_trimmed: bool,
    analysis_running: bool,
    on_clear_chat: Callable[[], None],
    logger,
) -> tuple[list[str], bool]:
    context_bar = st.container()
    selected_products: list[str] = []
    with context_bar:
        st.markdown('<div class="layout-zone layout-zone--inputs" data-context="true">', unsafe_allow_html=True)
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
            st.markdown("<p style='font-size:12px;font-weight:600;color:#0B2A4A;margin-bottom:4px;'>HTS Products (8- & 10-digit)</p>", unsafe_allow_html=True)
            if product_options:
                st.multiselect(
                    "HTS Products",
                    options=product_options,
                    key="selected_products_display_specific",
                    max_selections=max_products,
                    label_visibility="collapsed",
                    help="Pick HTS codes by exact 10-digit line or 8-digit category.",
                )
            else:
                st.error("No HTS products found — rebuild the SQLite database.")

        selected_countries = st.session_state.get("selected_countries", [])
        selected_labels = st.session_state.get("selected_products_display_specific", [])

        st.caption(
            f"{len(selected_countries)} / {max_country} countries · "
            f"{len(selected_labels)} HTS selections"
        )
        if country_trimmed:
            st.warning(f"Country selection limited to {max_country}. Extra choices were dropped.")
            logger.info("Trimmed country selection to limit", extra={"limit": max_country})
        if specific_trimmed:
            st.warning(f"Specific selections limited to {max_products}. Extra choices were dropped.")
            logger.info("Trimmed specific selection to limit", extra={"limit": max_products})

        action_cols = st.columns([1, 1], gap="large")
        selected_codes = [
            code_map[label] for label in selected_labels if label in code_map
        ]

        expanded_codes: list[str] = []
        seen_codes: set[str] = set()

        def _append_unique(value: str) -> None:
            if value and value not in seen_codes:
                expanded_codes.append(value)
                seen_codes.add(value)

        for code in selected_codes:
            _append_unique(code)

        final_trimmed = False
        if len(expanded_codes) > max_products:
            final_trimmed = True
            expanded_codes = expanded_codes[:max_products]

        if final_trimmed:
            st.warning(
                f"Using the first {max_products} codes due to the selection limit. "
                "Reduce your picks to refine the analysis."
            )
            logger.info(
                "Trimmed expanded specific selections to limit",
                extra={"limit": max_products},
            )

        st.session_state["selected_product_codes"] = expanded_codes
        selected_products = expanded_codes

        analyse_disabled = analysis_running or not selected_countries or not expanded_codes
        with action_cols[0]:
            analyse_clicked = st.button(
                "Analyze",
                disabled=analyse_disabled,
                width="stretch",
            )
        with action_cols[1]:
            if st.button("Clear Inputs", key="context_clear", width="stretch"):
                on_clear_chat()
        if analysis_running:
            st.caption("Running analysis…")
        st.markdown("</div>", unsafe_allow_html=True)
    return selected_products, analyse_clicked


def render_chat_composer(
    placeholder: str,
    *,
    pre_html: str | None = None,
    post_html: str | None = None,
) -> str | None:
    composer = st.container()
    with composer:
        st.markdown('<div class="layout-zone layout-zone--composer" data-composer="true">', unsafe_allow_html=True)
        st.markdown(
            """
            <style>
            [data-testid="stChatInputContainer"] {
                background:#F8FAFC !important;
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
                color:#6B7280 !important;
                opacity:1 !important;
            }
            [data-testid="stChatInputContainer"] button {
                background:var(--accent) !important;
                color:#ffffff !important;
                border:none !important;
                border-radius:999px !important;
                padding:0 18px !important;
                font-weight:600 !important;
                box-shadow:0 6px 18px rgba(11,42,74,0.3) !important;
            }
            [data-testid="stChatInputContainer"] button:hover {
                background:#0E3A6A !important;
            }
            [data-testid="stChatInputContainer"] button:focus-visible {
                outline:3px solid rgba(14,58,106,0.45) !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        if pre_html:
            st.markdown(pre_html, unsafe_allow_html=True)
        value = st.chat_input(placeholder, key="chat_input")
        if post_html:
            st.markdown(post_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return value


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
                    <span style='font-size:12px;font-weight:600;color:#031B4D;'>{snap.get('country','—')}</span>
                    <span style='font-size:10px;font-weight:600;padding:2px 8px;border-radius:999px;background:{badge_bg};color:{badge_color};text-transform:uppercase;'>{level}</span>
                </div>
                <div class='risk-gauge-track'>
                    <span style='left:0;width:33%;background:#1D9E75;'></span>
                    <span style='left:33%;width:34%;background:#BA7517;'></span>
                    <span style='left:67%;width:33%;background:#D85A30;'></span>
                    <div class='risk-gauge-needle' style='left:{_needle(score)}%;'></div>
                </div>
                <div style='display:flex;justify-content:space-between;font-size:9px;color:#64748B;margin-top:2px;text-transform:uppercase;letter-spacing:0.04em;'>
                    <span>Low</span><span>Medium</span><span>High</span>
                </div>
                <div style='margin-top:8px;display:flex;align-items:flex-end;gap:6px;'>
                    <span style='font-family:monospace;font-size:20px;font-weight:600;color:#0B2A4A;'>{score}</span>
                    <span style='font-size:11px;color:#6B7280;'>/ 100</span>
                </div>
            </div>
            """
        )
    st.markdown(
        f"<div class='risk-gauge-grid'>{''.join(cards)}</div>",
        unsafe_allow_html=True,
    )


def render_chat_feed(
    messages: list[dict],
    latest_result: dict | None,
    *,
    code_lookup: dict[str, str] | None = None,
    chart_mode: str = "ad_valorem",
    render_chart: Callable[[pd.DataFrame, str], go.Figure | None] | None = None,
):
    chat_feed = st.container()
    with chat_feed:
        st.markdown('<div class="layout-zone layout-zone--chat" data-chat="true">', unsafe_allow_html=True)
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
                    readable = []
                    for raw_code in selections['products']:
                        if code_lookup:
                            display = code_lookup.get(raw_code)
                        else:
                            display = None
                        if display:
                            readable.append(display.split(' — ', 1)[-1])
                        else:
                            readable.append(raw_code)
                    selection_text.append('Products: ' + ', '.join(readable))
                if selection_text:
                    st.markdown(
                        f"<div class='analysis-meta'>{' · '.join(selection_text)}</div>",
                        unsafe_allow_html=True,
                    )
                fig_payload = msg.get('plotly_fig')
                raw_data = msg.get('raw_chart_data')
                raw_cols = msg.get('raw_chart_columns')
                fig = None
                if render_chart and raw_data and raw_cols:
                    try:
                        chart_df = pd.DataFrame(raw_data, columns=raw_cols)
                        fig = render_chart(chart_df, chart_mode)
                    except Exception:
                        fig = None
                if fig is None and fig_payload:
                    fig = go.Figure(fig_payload)
                if fig:
                    st.plotly_chart(fig, width="stretch")
                    if render_chart and raw_data and raw_cols:
                        if chart_mode == "ad_valorem":
                            st.caption(
                                "Orange ring = effective rate modified by a Chapter 99 surcharge or trade program override."
                            )
                        else:
                            st.caption(
                                "Y axis shows the specific duty amount from the HTS table. "
                                "Ch.99 percentage surcharges are not applied in this view."
                            )
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
                    st.dataframe(chart_df, width="stretch")
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
                    width="stretch",
                )
            else:
                st.info('The last query returned no rows.')
        st.markdown('<div data-anchor="chat-end" id="chat-end"></div>', unsafe_allow_html=True)
        if analysis_stream_placeholder is None:
            analysis_stream_placeholder = st.empty()
    render_inline_iframe(
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
    st.markdown("</div>", unsafe_allow_html=True)
    return analysis_stream_placeholder
