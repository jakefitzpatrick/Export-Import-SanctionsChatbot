"""Streamlit text-to-SQL chatbot backed by a local HTS SQLite database."""
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
import openai
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from risk_model import get_risk_df
from session import (
    MAX_COUNTRY_SELECTION,
    MAX_PRODUCT_OPTIONS,
    MAX_PRODUCT_SELECTION,
    DEFAULT_COUNTRY_SELECTION,
    enforce_selection_limit,
    reset_app_state,
)
from utils import LAST_RESULT_KEY
from chat import (
    append_message,
    answer_question,
    QUESTION_PLACEHOLDER,
)
from analysis import (
    queue_analysis_request,
    maybe_run_analysis,
    render_correlation_chart,
)
from logger import setup_logger
from app_ui import (
    get_css,
    render_sidebar,
)

load_dotenv()

logger = setup_logger(__name__)

HTS_DB_PATH = Path(__file__).resolve().parent / "data" / "hts.db"


def _handle_sidebar_clear() -> None:
    """Reset chat selections via sidebar action."""
    reset_app_state()
    logger.info("Chat cleared from sidebar button")
    st.rerun()


@st.cache_resource
def get_db_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(show_spinner=False)
def load_product_options(_conn: sqlite3.Connection) -> list[tuple[str, str]]:
    query = (
        'SELECT hts_code, description FROM hts '
        'WHERE hts_code IS NOT NULL AND hts_code <> "" '
        'ORDER BY hts_code LIMIT ?'
    )
    logger.info("Loading product options from SQLite", extra={"limit": MAX_PRODUCT_OPTIONS})
    try:
        df = pd.read_sql_query(query, _conn, params=(MAX_PRODUCT_OPTIONS,))
    except Exception as exc:
        logger.warning("Failed to load product options: %s", exc)
        return []
    logger.info("Loaded %s product descriptions into the picker", len(df))
    return list(df.itertuples(index=False, name=None))


# ---------------------------------------------------------------------------
# Trade-flow map
# ---------------------------------------------------------------------------

# Country name (as used in the app/HTS data) → (ISO-3 code, latitude, longitude)
# Includes all names from the risk-model dropdown AND the Ch99 database.
# Aliases (same ISO, different name) are intentional — choropleth deduplicates by ISO.
COUNTRY_GEO: dict[str, tuple[str, float, float]] = {
    "Afghanistan": ("AFG", 33.9, 67.7), "Albania": ("ALB", 41.2, 20.2),
    "Algeria": ("DZA", 28.0, 1.7), "Andorra": ("AND", 42.5, 1.5),
    "Angola": ("AGO", -11.2, 17.9), "Antigua and Barbuda": ("ATG", 17.1, -61.8),
    "Argentina": ("ARG", -38.4, -63.6), "Armenia": ("ARM", 40.1, 45.0),
    "Australia": ("AUS", -25.0, 133.0), "Austria": ("AUT", 47.5, 14.6),
    "Azerbaijan": ("AZE", 40.1, 47.6), "Bahamas": ("BHS", 25.0, -77.3),
    "Bahrain": ("BHR", 26.0, 50.6), "Bangladesh": ("BGD", 24.0, 90.0),
    "Barbados": ("BRB", 13.2, -59.6), "Belarus": ("BLR", 53.7, 28.0),
    "Belgium": ("BEL", 50.8, 4.5), "Belize": ("BLZ", 17.2, -88.5),
    "Benin": ("BEN", 9.3, 2.3), "Bhutan": ("BTN", 27.5, 90.4),
    "Bolivia": ("BOL", -16.3, -64.5), "Bosnia and Herzegovina": ("BIH", 44.2, 17.8),
    "Botswana": ("BWA", -22.3, 24.7), "Brazil": ("BRA", -14.2, -51.9),
    "Brunei": ("BRN", 4.5, 114.7), "Bulgaria": ("BGR", 42.7, 25.5),
    "Burkina Faso": ("BFA", 12.4, -1.6),
    "Burma": ("MMR", 17.1, 96.0),                  # Ch99 spelling
    "Burma/Myanmar": ("MMR", 17.1, 96.0),           # risk-model spelling
    "Myanmar": ("MMR", 17.1, 96.0),                 # alternate spelling
    "Burundi": ("BDI", -3.4, 30.0), "Cambodia": ("KHM", 12.6, 104.9),
    "Cameroon": ("CMR", 5.7, 12.4), "Canada": ("CAN", 60.0, -96.0),
    "Cape Verde": ("CPV", 15.1, -23.6), "Central African Republic": ("CAF", 6.6, 20.9),
    "Chad": ("TCD", 15.5, 18.7), "Chile": ("CHL", -35.7, -71.5),
    "China": ("CHN", 35.9, 104.2), "Colombia": ("COL", 4.1, -72.3),
    "Comoros": ("COM", -11.6, 43.3), "Costa Rica": ("CRI", 9.7, -83.8),
    "Cote d'Ivoire": ("CIV", 7.5, -5.5),            # Ch99 spelling
    "Ivory Coast": ("CIV", 7.5, -5.5),              # risk-model spelling
    "Croatia": ("HRV", 45.1, 15.2),
    "Cyprus": ("CYP", 35.1, 33.4), "Czech Republic": ("CZE", 49.8, 15.5),
    "DR Congo": ("COD", -4.0, 21.8),                # Ch99 spelling
    "Democratic Republic of the Congo": ("COD", -4.0, 21.8),  # risk-model spelling
    "Republic of the Congo": ("COG", -0.2, 15.8),   # risk-model (different country)
    "Denmark": ("DNK", 56.3, 9.5), "Djibouti": ("DJI", 11.8, 42.6),
    "Dominica": ("DMA", 15.4, -61.4), "Dominican Republic": ("DOM", 18.7, -70.2),
    "Ecuador": ("ECU", -1.8, -78.2), "Egypt": ("EGY", 26.8, 30.8),
    "El Salvador": ("SLV", 13.8, -88.9), "Equatorial Guinea": ("GNQ", 1.7, 10.3),
    "Eritrea": ("ERI", 15.2, 39.8), "Estonia": ("EST", 58.6, 25.0),
    "Ethiopia": ("ETH", 9.1, 40.5), "Falkland Islands": ("FLK", -51.8, -59.5),
    "Fiji": ("FJI", -17.7, 178.1), "Finland": ("FIN", 64.0, 26.0),
    "France": ("FRA", 46.2, 2.2), "Gabon": ("GAB", -0.8, 11.6),
    "Gambia": ("GMB", 13.4, -15.3),                 # Ch99 spelling
    "The Gambia": ("GMB", 13.4, -15.3),             # risk-model spelling
    "Georgia": ("GEO", 42.3, 43.4), "Germany": ("DEU", 51.2, 10.5),
    "Ghana": ("GHA", 7.9, -1.0), "Greece": ("GRC", 39.1, 21.8),
    "Grenada": ("GRD", 12.1, -61.7), "Guatemala": ("GTM", 15.8, -90.2),
    "Guinea": ("GIN", 11.0, -10.9), "Guinea-Bissau": ("GNB", 11.8, -15.2),
    "Guyana": ("GUY", 4.9, -58.9), "Haiti": ("HTI", 19.0, -72.3),
    "Honduras": ("HND", 15.2, -86.2), "Hong Kong": ("HKG", 22.3, 114.2),
    "Hungary": ("HUN", 47.2, 19.5), "Iceland": ("ISL", 64.9, -18.7),
    "India": ("IND", 20.6, 79.0), "Indonesia": ("IDN", -0.8, 113.9),
    "Iran": ("IRN", 32.4, 53.7), "Iraq": ("IRQ", 33.2, 43.7),
    "Ireland": ("IRL", 53.4, -8.2), "Israel": ("ISR", 31.0, 35.0),
    "Italy": ("ITA", 41.9, 12.6), "Jamaica": ("JAM", 18.1, -77.3),
    "Japan": ("JPN", 36.2, 138.3), "Jordan": ("JOR", 30.6, 36.2),
    "Kazakhstan": ("KAZ", 48.0, 67.3), "Kenya": ("KEN", -0.1, 37.9),
    "Kiribati": ("KIR", -3.4, -168.7), "Kosovo": ("XKX", 42.6, 20.9),
    "Kuwait": ("KWT", 29.3, 47.5), "Kyrgyzstan": ("KGZ", 41.2, 74.8),
    "Laos": ("LAO", 19.9, 102.5), "Latvia": ("LVA", 56.9, 24.6),
    "Lebanon": ("LBN", 33.9, 35.9), "Lesotho": ("LSO", -29.6, 28.2),
    "Liberia": ("LBR", 6.4, -9.4), "Libya": ("LBY", 26.3, 17.2),
    "Liechtenstein": ("LIE", 47.2, 9.5), "Lithuania": ("LTU", 55.2, 24.0),
    "Luxembourg": ("LUX", 49.8, 6.1), "Macau": ("MAC", 22.2, 113.5),
    "Macedonia": ("MKD", 41.6, 21.7), "North Macedonia": ("MKD", 41.6, 21.7),
    "Madagascar": ("MDG", -18.8, 46.9), "Malawi": ("MWI", -13.3, 34.3),
    "Malaysia": ("MYS", 4.2, 108.0), "Maldives": ("MDV", 3.2, 73.2),
    "Mali": ("MLI", 17.6, -4.0), "Malta": ("MLT", 35.9, 14.4),
    "Marshall Islands": ("MHL", 7.1, 171.2), "Mauritania": ("MRT", 21.0, -10.9),
    "Mauritius": ("MUS", -20.3, 57.6), "Mexico": ("MEX", 23.6, -102.6),
    "Micronesia": ("FSM", 7.4, 150.6), "Moldova": ("MDA", 47.4, 28.4),
    "Monaco": ("MCO", 43.7, 7.4), "Mongolia": ("MNG", 46.9, 103.8),
    "Montenegro": ("MNE", 42.7, 19.4), "Morocco": ("MAR", 31.8, -7.1),
    "Mozambique": ("MOZ", -18.7, 35.5), "Namibia": ("NAM", -22.9, 18.5),
    "Nauru": ("NRU", -0.5, 166.9), "Nepal": ("NPL", 28.4, 84.1),
    "Netherlands": ("NLD", 52.1, 5.3), "New Zealand": ("NZL", -40.9, 174.9),
    "Nicaragua": ("NIC", 12.9, -85.2), "Niger": ("NER", 17.6, 8.1),
    "Nigeria": ("NGA", 9.1, 8.7), "North Korea": ("PRK", 40.3, 127.5),
    "Norway": ("NOR", 60.5, 8.5), "Oman": ("OMN", 21.5, 55.9),
    "Pakistan": ("PAK", 30.4, 69.3), "Palau": ("PLW", 7.5, 134.6),
    "Palestine/West Bank": ("PSE", 31.9, 35.2), "Panama": ("PAN", 8.4, -80.1),
    "Papua New Guinea": ("PNG", -6.3, 143.9), "Paraguay": ("PRY", -23.4, -58.4),
    "Peru": ("PER", -9.2, -75.0), "Philippines": ("PHL", 12.9, 121.8),
    "Poland": ("POL", 51.9, 19.1), "Portugal": ("PRT", 39.4, -8.2),
    "Qatar": ("QAT", 25.4, 51.2), "Romania": ("ROU", 45.9, 24.9),
    "Russia": ("RUS", 61.5, 105.3), "Rwanda": ("RWA", -1.9, 29.9),
    "Saint Kitts and Nevis": ("KNA", 17.4, -62.8), "Saint Lucia": ("LCA", 13.9, -60.9),
    "Saint Vincent and the Grenadines": ("VCT", 13.3, -61.2),
    "Samoa": ("WSM", -13.8, -172.1), "San Marino": ("SMR", 43.9, 12.5),
    "Sao Tome and Principe": ("STP", 0.2, 6.6), "Saudi Arabia": ("SAU", 24.0, 45.0),
    "Senegal": ("SEN", 14.5, -14.5), "Serbia": ("SRB", 44.0, 21.0),
    "Seychelles": ("SYC", -4.7, 55.5), "Sierra Leone": ("SLE", 8.5, -11.8),
    "Singapore": ("SGP", 1.4, 103.8), "Slovakia": ("SVK", 48.7, 19.7),
    "Slovenia": ("SVN", 46.2, 15.0), "Solomon Islands": ("SLB", -9.6, 160.2),
    "Somalia": ("SOM", 5.2, 46.2), "South Africa": ("ZAF", -30.6, 22.9),
    "South Korea": ("KOR", 35.9, 127.8), "South Sudan": ("SSD", 6.9, 31.3),
    "Spain": ("ESP", 40.5, -3.7), "Sri Lanka": ("LKA", 7.9, 80.8),
    "Sudan": ("SDN", 12.9, 30.2), "Suriname": ("SUR", 3.9, -56.0),
    "Swaziland": ("SWZ", -26.5, 31.5), "Sweden": ("SWE", 60.1, 18.6),
    "Switzerland": ("CHE", 47.0, 8.2), "Syria": ("SYR", 34.8, 38.9),
    "Taiwan": ("TWN", 23.7, 120.9), "Tanzania": ("TZA", -6.4, 34.9),
    "Thailand": ("THA", 15.9, 100.9), "Timor-Leste": ("TLS", -8.9, 125.7),
    "Togo": ("TGO", 8.6, 0.8), "Tonga": ("TON", -21.2, -175.2),
    "Trinidad and Tobago": ("TTO", 10.7, -61.2), "Tunisia": ("TUN", 33.9, 9.6),
    "Turkey": ("TUR", 38.6, 35.2),                  # Ch99 spelling
    "Türkiye": ("TUR", 38.6, 35.2),                 # risk-model spelling
    "Tuvalu": ("TUV", -8.5, 179.2), "Uganda": ("UGA", 1.4, 32.3),
    "Ukraine": ("UKR", 48.4, 31.2), "United Arab Emirates": ("ARE", 23.4, 53.8),
    "United Kingdom": ("GBR", 55.4, -3.4),
    "United States": ("USA", 39.5, -98.4),           # Ch99 spelling
    "United States of America": ("USA", 39.5, -98.4),  # risk-model spelling
    "Uruguay": ("URY", -32.5, -55.8), "Uzbekistan": ("UZB", 41.4, 64.6),
    "Vanuatu": ("VUT", -15.4, 166.9), "Venezuela": ("VEN", 6.4, -66.6),
    "Vietnam": ("VNM", 14.1, 108.3), "Yemen": ("YEM", 15.6, 48.5),
    "Zambia": ("ZMB", -13.1, 27.9), "Zimbabwe": ("ZWE", -20.0, 30.0),
}

_USA_LAT, _USA_LON = 39.5, -98.4


def _slerp_arc(
    lat1: float, lon1: float, lat2: float, lon2: float, n: int = 80
) -> tuple[list[float], list[float]]:
    """Return (lats, lons) for a great-circle arc from point 1 to point 2."""
    r = np.radians
    def to_xyz(la, lo):
        return np.array([np.cos(r(la)) * np.cos(r(lo)),
                         np.cos(r(la)) * np.sin(r(lo)),
                         np.sin(r(la))])
    v1, v2 = to_xyz(lat1, lon1), to_xyz(lat2, lon2)
    omega = np.arccos(float(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    lats, lons = [], []
    for t in np.linspace(0.0, 1.0, n):
        if omega < 1e-10:
            v = v1
        else:
            v = (np.sin((1 - t) * omega) * v1 + np.sin(t * omega) * v2) / np.sin(omega)
        lats.append(float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0)))))
        lons.append(float(np.degrees(np.arctan2(v[1], v[0]))))
    return lats, lons


@st.cache_data(ttl=60)
def render_trade_map(selected_countries: tuple[str, ...]) -> go.Figure:
    """Choropleth with animated comet-trail arcs flowing from selected countries to USA."""
    N_FRAMES = 50
    N_ARC = 80
    TAIL = 10  # comet tail length in points

    sel_set = set(selected_countries)
    _USA_ISO = "USA"

    iso_z: dict[str, int] = {}
    for name, (iso, _la, _lo) in COUNTRY_GEO.items():
        if iso == _USA_ISO:
            z = 2
        elif name in sel_set:
            z = 1
        else:
            z = 0
        iso_z[iso] = max(iso_z.get(iso, 0), z)

    iso_list = list(iso_z.keys())
    z_list = list(iso_z.values())

    arcs: dict[str, tuple[list[float], list[float]]] = {}
    for country in selected_countries:
        geo = COUNTRY_GEO.get(country)
        if geo and geo[0] != _USA_ISO:
            arcs[country] = _slerp_arc(geo[1], geo[2], _USA_LAT, _USA_LON, N_ARC)

    arc_list = list(arcs.items())

    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        locations=iso_list,
        z=z_list,
        locationmode="ISO-3",
        colorscale=[
            [0.0, "#1e2030"],
            [0.5, "#e8733a"],
            [1.0, "#2b7de9"],
        ],
        zmin=0, zmax=2,
        showscale=False,
        marker_line_width=0.4,
        marker_line_color="#3a3d52",
        hovertemplate="%{location}<extra></extra>",
    ))

    for _country, (lats, lons) in arc_list:
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons,
            mode="lines",
            line=dict(width=1.2, color="rgba(255,255,255,0.18)", dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))

    dot_trace_indices = []
    for _country, (lats, lons) in arc_list:
        dot_trace_indices.append(len(fig.data))
        fig.add_trace(go.Scattergeo(
            lat=[lats[0]], lon=[lons[0]],
            mode="markers",
            marker=dict(size=8, color="#ffcc44", opacity=1.0),
            hoverinfo="skip", showlegend=False,
        ))

    fig.add_trace(go.Scattergeo(
        lat=[_USA_LAT], lon=[_USA_LON],
        mode="markers",
        marker=dict(size=14, color="#2b7de9", symbol="star",
                    line=dict(width=1.5, color="white")),
        hoverinfo="skip", showlegend=False,
    ))

    frames = []
    for f in range(N_FRAMES):
        t = f / N_FRAMES
        head_idx = int(t * (N_ARC - 1))
        frame_data = []
        for _country, (lats, lons) in arc_list:
            tail_start = max(0, head_idx - TAIL + 1)
            t_lats = lats[tail_start: head_idx + 1]
            t_lons = lons[tail_start: head_idx + 1]
            n_tail = len(t_lats)
            sizes = [3 + 7 * (i / max(n_tail - 1, 1)) for i in range(n_tail)]
            alphas = [0.15 + 0.85 * (i / max(n_tail - 1, 1)) for i in range(n_tail)]
            colors = [f"rgba(255,204,68,{a:.2f})" for a in alphas]
            frame_data.append(go.Scattergeo(
                lat=t_lats, lon=t_lons,
                mode="markers",
                marker=dict(size=sizes, color=colors, symbol="circle"),
                hoverinfo="skip",
            ))
        frames.append(go.Frame(data=frame_data, traces=dot_trace_indices))

    fig.frames = frames

    play_menu = dict(
        type="buttons", showactive=False,
        x=1.05, y=0.5,
        xanchor="left", yanchor="middle",
        buttons=[dict(
            label="▶",
            method="animate",
            args=[None, dict(
                frame=dict(duration=45, redraw=False),
                fromcurrent=False,
                transition=dict(duration=0),
                mode="immediate",
            )],
        )],
    )

    fig.update_layout(
        paper_bgcolor="#0f1117",
        margin=dict(l=0, r=0, t=4, b=0),
        height=260,
        geo=dict(
            bgcolor="#0f1117",
            showland=True, landcolor="#1e2030",
            showocean=True, oceancolor="#12141f",
            showcoastlines=True, coastlinecolor="#2e3148",
            showcountries=True, countrycolor="#2e3148",
            showframe=False,
            projection_type="natural earth",
            lataxis_range=[-60, 85],
        ),
        updatemenus=[play_menu] if arc_list else [],
    )

    return fig


def main() -> None:
    st.set_page_config(page_title="ImportInsight AI", layout="wide")
    st.markdown(get_css(), unsafe_allow_html=True)
    logger.info("Streamlit UI initialized")

    openai.api_type = "azure"
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        st.error("Please set AZURE_OPENAI_API_KEY before running the app.")
        logger.error("AZURE_OPENAI_API_KEY missing; blocking startup")
        return
    if not openai.api_base:
        st.error("Please set AZURE_OPENAI_ENDPOINT before running the app.")
        logger.error("AZURE_OPENAI_ENDPOINT missing; blocking startup")
        return
    openai.api_key = api_key

    deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")
    if not deployment_id:
        st.error("Please set AZURE_OPENAI_DEPLOYMENT_ID for the chat completion deployment.")
        logger.error("AZURE_OPENAI_DEPLOYMENT_ID missing; blocking startup")
        return

    if not HTS_DB_PATH.exists():
        st.error(
            "The local HTS database is missing. Run `python scripts/build_hts_sqlite.py` to create data/hts.db before launching the app."
        )
        logger.error("hts.db missing at %s", HTS_DB_PATH)
        return

    conn = get_db_connection(str(HTS_DB_PATH))
    logger.info("Connected to SQLite database", extra={"path": str(HTS_DB_PATH)})

    risk_df = get_risk_df()
    country_options = sorted(risk_df["country"].tolist(), key=str.lower)
    if not country_options:
        country_options = DEFAULT_COUNTRY_SELECTION.copy()
    product_options = load_product_options(conn)
    display_options = [f"{code} — {desc}" for code, desc in product_options]
    code_map = dict(zip(display_options, [code for code, _ in product_options]))
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.setdefault(LAST_RESULT_KEY, None)
    if "selected_countries" not in st.session_state:
        st.session_state["selected_countries"] = (
            country_options[:MAX_COUNTRY_SELECTION] or DEFAULT_COUNTRY_SELECTION.copy()
        )
    if "selected_products_display" not in st.session_state:
        st.session_state["selected_products_display"] = (
            display_options[: min(MAX_PRODUCT_SELECTION, len(display_options))] if display_options else []
        )
    st.session_state.setdefault("selected_product_codes", [])
    st.session_state.setdefault("analysis_inflight", False)
    st.session_state.setdefault("analysis_active_run", None)
    st.session_state.setdefault("analysis_request", None)
    st.session_state.setdefault("correlation_signature", None)
    st.session_state.setdefault("chat_scroll_token", 0)
    st.session_state.setdefault("chart_mode", "ad_valorem")

    selected_countries, country_trimmed = enforce_selection_limit(
        "selected_countries",
        MAX_COUNTRY_SELECTION,
    )
    selected_products_display, product_trimmed = enforce_selection_limit(
        "selected_products_display",
        MAX_PRODUCT_SELECTION,
    )

    render_sidebar(_handle_sidebar_clear)

    context_bar = st.container()
    with context_bar:
        st.markdown('<div data-context="true"></div>', unsafe_allow_html=True)
        st.markdown("### Context")
        st.caption("Select up to three countries and products to drive the analysis below.")

        # Trade-flow map — updates live as countries are chosen
        _map_countries = tuple(st.session_state.get("selected_countries", []))
        _map_fig = render_trade_map(_map_countries)
        st.plotly_chart(
            _map_fig,
            use_container_width=True,
            config={"staticPlot": False, "displayModeBar": False},
            key="trade_map",
        )
        if _map_countries:
            components.html(
                """<script>
                (function tryPlay() {
                    var attempts = 0;
                    function attempt() {
                        attempts++;
                        try {
                            var iframes = window.parent.document.querySelectorAll('iframe[title="trade_map"]');
                            if (!iframes.length) iframes = window.parent.document.querySelectorAll('.stPlotlyChart iframe');
                            for (var i = 0; i < iframes.length; i++) {
                                var inner = iframes[i].contentWindow || iframes[i].contentDocument.defaultView;
                                var graphs = inner.document.querySelectorAll('.js-plotly-plot');
                                graphs.forEach(function(g) {
                                    if (g._fullLayout && g._fullLayout.updatemenus && g._fullLayout.updatemenus.length) {
                                        Plotly.animate(g, null, {
                                            frame: {duration: 45, redraw: false},
                                            transition: {duration: 0},
                                            mode: 'immediate',
                                        });
                                    }
                                });
                            }
                        } catch(e) {}
                        if (attempts < 20) setTimeout(attempt, 300);
                    }
                    setTimeout(attempt, 500);
                })();
                </script>""",
                height=0,
            )

        sel_cols = st.columns(2)
        with sel_cols[0]:
            st.multiselect(
                "Countries",
                options=country_options,
                key="selected_countries",
                max_selections=MAX_COUNTRY_SELECTION,
                help="Choose up to three countries to compare governance risk.",
            )
        with sel_cols[1]:
            if display_options:
                st.multiselect(
                    "HTS Products",
                    options=display_options,
                    key="selected_products_display",
                    max_selections=MAX_PRODUCT_SELECTION,
                    help="Products are loaded from the local HTS SQLite database.",
                )
            else:
                st.error("No HTS products found—rebuild the SQLite database.")
        selected_countries = st.session_state.get("selected_countries", [])
        selected_products_display = st.session_state.get("selected_products_display", [])
        st.caption(
            f"{len(selected_countries)} / {MAX_COUNTRY_SELECTION} countries · {len(selected_products_display)} / {MAX_PRODUCT_SELECTION} products"
        )
        if country_trimmed:
            st.warning(
                f"Country selection limited to {MAX_COUNTRY_SELECTION}. Extra choices were dropped."
            )
            logger.info("Trimmed country selection to limit", extra={"limit": MAX_COUNTRY_SELECTION})
        if product_trimmed:
            st.warning(
                f"Product selection limited to {MAX_PRODUCT_SELECTION}. Extra choices were dropped."
            )
            logger.info("Trimmed product selection to limit", extra={"limit": MAX_PRODUCT_SELECTION})
        current_product_labels = st.session_state.get("selected_products_display", [])
        valid_product_labels = [label for label in current_product_labels if label in code_map]
        if len(valid_product_labels) != len(current_product_labels):
            st.warning("Some selected products are unavailable in the current HTS list.")
        selected_products = [code_map[label] for label in valid_product_labels]
        st.session_state["selected_product_codes"] = selected_products

        action_cols = st.columns([1, 1])
        analyse_disabled = (
            st.session_state.get("analysis_inflight")
            or not selected_countries
            or not selected_products
        )
        with action_cols[0]:
            analyse_clicked = st.button(
                "Analyse",
                width="stretch",
                disabled=analyse_disabled,
            )
        with action_cols[1]:
            if st.button("Clear Chat", width="stretch", key="context_clear"):
                reset_app_state()
                st.rerun()
        if analyse_clicked:
            queued = queue_analysis_request(selected_countries, selected_products)
            if not queued:
                st.warning("Select at least one country and one product before running analysis.")
                logger.warning(
                    "Analyse button pressed without valid selections",
                    extra={"countries": selected_countries, "products": selected_products},
                )
        if st.session_state.get("analysis_inflight"):
            st.caption("Running analysis…")

        # Chart mode toggle — persists across reruns so switching is instant
        _last_analysis = next(
            (m for m in reversed(st.session_state.get("messages", []))
             if m.get("type") == "analysis"),
            None,
        )
        _has_specific = bool(_last_analysis and _last_analysis.get("has_specific_data"))
        _toggle_help = (
            "Switch the Y axis between the ad valorem percentage rate and the specific duty amount (e.g. $/kg, ¢/dozen)."
            if _has_specific
            else "This product has a percentage-based rate only. Select a product with a unit-based rate (e.g. $/kg, ¢/liter) to use this view."
        )
        if not _has_specific:
            st.session_state["chart_mode"] = "ad_valorem"
        show_specific = st.toggle(
            "Show specific duty rate ($/unit)",
            value=(st.session_state["chart_mode"] == "specific"),
            disabled=not _has_specific,
            help=_toggle_help,
        )
        if _has_specific:
            st.session_state["chart_mode"] = "specific" if show_specific else "ad_valorem"

    analysis_stream_placeholder: st.delta_generator.DeltaGenerator | None = None
    chat_feed = st.container()
    composer = st.container()

    with composer:
        st.markdown('<div data-composer="true"></div>', unsafe_allow_html=True)
        prompt = st.chat_input(QUESTION_PLACEHOLDER, key="chat_input")

    if prompt is not None:
        question = prompt.strip()
        if not question:
            st.warning("Please enter a prompt before sending.")
        else:
            timestamp = datetime.now().strftime("%I:%M %p")
            logger.info("User submitted prompt", extra={"prompt": question})
            append_message({"role": "user", "content": question, "time": timestamp})
            try:
                assistant_text, metadata = answer_question(question, deployment_id, conn)
                logger.info(
                    "Assistant response ready",
                    extra={"mode": metadata.get("mode"), "sql_rows": metadata.get("row_count")},
                )
                if metadata.get("mode") == "sql":
                    assistant_text = (
                        f"{assistant_text}\n\n_(Ran a fresh SQL query; see \"Last SQL Result\" below.)_"
                    )
            except Exception as exc:
                logger.exception("Chat response failed")
                assistant_text = f"Error while answering your question: {exc}"
            append_message(
                {"role": "assistant", "content": assistant_text, "time": datetime.now().strftime("%I:%M %p")}
            )

    with chat_feed:
        st.markdown('<div data-chat="true"></div>', unsafe_allow_html=True)
        if len(st.session_state.messages) == 0:
            st.markdown(
                "<div class='empty-chat'>Select countries/products above or ask a question about the HTS data.</div>",
                unsafe_allow_html=True,
            )
        for msg in st.session_state.messages:
            timestamp = msg.get("time", "")
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='bubble-label bubble-label-right'>You</div><div class='bubble-user'>{msg['content']}</div><div class='timestamp timestamp-right'>{timestamp}</div><div class='clearfix'></div>",
                    unsafe_allow_html=True,
                )
                continue

            if msg.get("type") == "analysis":
                st.markdown(
                    "<div class='bubble-label'>Assistant</div>",
                    unsafe_allow_html=True,
                )
                selections = msg.get("selections", {})
                selection_text = []
                if selections.get("countries"):
                    selection_text.append("Countries: " + ", ".join(selections["countries"]))
                if selections.get("products"):
                    readable = []
                    for raw_code in selections["products"]:
                        match = next(
                            (lbl for lbl, code in code_map.items() if code == raw_code),
                            None,
                        )
                        readable.append(match.split(" — ", 1)[-1] if match else raw_code)
                    selection_text.append("Products: " + ", ".join(readable))
                if selection_text:
                    st.markdown(
                        f"<div class='analysis-meta'>{' · '.join(selection_text)}</div>",
                        unsafe_allow_html=True,
                    )

                # Scatterplot — rebuilt live from raw data so toggle takes effect instantly
                _raw_data = msg.get("raw_chart_data")
                _raw_cols = msg.get("raw_chart_columns")
                _chart_data = msg.get("chart_data")
                _chart_cols = msg.get("chart_columns")

                if _raw_data and _raw_cols:
                    # New-style message with raw data for toggle re-rendering
                    _chart_df = pd.DataFrame(_raw_data, columns=_raw_cols)
                    _mode = st.session_state.get("chart_mode", "ad_valorem")
                    fig = render_correlation_chart(_chart_df, mode=_mode)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        if _mode == "ad_valorem":
                            st.caption(
                                "Orange ring = effective rate modified by a Chapter 99 surcharge or trade program override."
                            )
                        else:
                            st.caption(
                                "Y axis shows the specific duty amount from the HTS table. "
                                "Ch.99 percentage surcharges are not applied in this view."
                            )
                elif msg.get("plotly_fig"):
                    # Fallback: render stored static figure
                    fig = go.Figure(msg["plotly_fig"])
                    st.plotly_chart(fig, use_container_width=True)

                headline = msg.get("headline")
                body_html = msg["content"]
                if headline:
                    body_html = f"<p class='analysis-headline'><strong>{headline}</strong></p>{body_html}"
                st.markdown(
                    f"<div class='bubble-bot'>{body_html}</div><div class='timestamp'>{timestamp}</div><div class='clearfix'></div>",
                    unsafe_allow_html=True,
                )
                non_ad_summary_text = msg.get("non_ad_summary_text")
                if non_ad_summary_text:
                    st.info(non_ad_summary_text)
                risk_snapshot = msg.get("risk_snapshot") or []
                if risk_snapshot:
                    pills_html = "".join(
                        f"<div class='risk-pill' style='border-left-color:{snap['color']};'>"
                        f"<strong>{snap['country']}</strong>"
                        f"Score {snap['score']:.1f} · {snap['level']}"
                        "</div>"
                        for snap in risk_snapshot
                    )
                    st.markdown(f"<div class='risk-pills'>{pills_html}</div>", unsafe_allow_html=True)
                # Display table — use formatted display data if available
                if _chart_data and _chart_cols:
                    chart_df = pd.DataFrame(_chart_data, columns=_chart_cols)
                    st.dataframe(chart_df)
                continue

            st.markdown(
                f"<div class='bubble-label'>Assistant</div><div class='bubble-bot'>{msg['content']}</div><div class='timestamp'>{timestamp}</div><div class='clearfix'></div>",
                unsafe_allow_html=True,
            )

        analysis_stream_placeholder = st.empty()
        latest_result = st.session_state.get(LAST_RESULT_KEY)
        if latest_result:
            st.markdown("---")
            st.markdown("### Last SQL Result")
            st.code(latest_result["sql"], language="sql")
            st.caption(
                f"Returned {latest_result['row_count']} row(s); showing {latest_result['rows_displayed']} row(s) below."
            )
            if latest_result["records"]:
                st.dataframe(
                    pd.DataFrame(latest_result["records"], columns=latest_result["columns"])
                )
            else:
                st.info("The last query returned no rows.")
        st.markdown('<div data-anchor="chat-end" id="chat-end"></div>', unsafe_allow_html=True)
        if analysis_stream_placeholder is None:
            analysis_stream_placeholder = st.empty()
        maybe_run_analysis(
            conn,
            deployment_id,
            risk_df,
            analysis_stream_placeholder,
        )

    components.html(
        f"""
        <script>
        const marker = window.parent.document.querySelector('div[data-anchor="chat-end"]');
        if (marker) {{
            const chatBlock = marker.closest('div[data-testid="stVerticalBlock"]');
            if (chatBlock) {{
                chatBlock.scrollTop = chatBlock.scrollHeight;
            }}
        }}
        </script>
        """,
        height=0,
    )

if __name__ == "__main__":
    main()
