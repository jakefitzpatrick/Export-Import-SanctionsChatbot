"""Session-state helpers and configuration."""
from __future__ import annotations

from typing import Sequence

import pandas as pd
import streamlit as st

from utils import LAST_RESULT_KEY, LLM_CACHE_KEY

# Allow the picker to load the full HTS list (~23k rows) while keeping a hard ceiling.
MAX_PRODUCT_OPTIONS = 50000
MAX_COUNTRY_SELECTION = 3
MAX_PRODUCT_SELECTION = 1
DEFAULT_COUNTRY_SELECTION = ["Cameroon", "Russia"]


def bootstrap_session_state(
    country_options: list[str],
    display_options: list[str],
) -> None:
    """Ensure all Streamlit session keys exist with defaults."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.setdefault(LAST_RESULT_KEY, None)
    st.session_state.setdefault(
        "selected_countries",
        country_options[:MAX_COUNTRY_SELECTION] or DEFAULT_COUNTRY_SELECTION.copy(),
    )
    st.session_state.setdefault("product_mode", "specific")
    st.session_state.setdefault("selected_products_display_specific", [])
    st.session_state.setdefault("selected_products_display_categories", [])
    st.session_state.setdefault("selected_product_codes", [])
    st.session_state.setdefault("analysis_inflight", False)
    st.session_state.setdefault("analysis_active_run", None)
    st.session_state.setdefault("correlation_signature", None)
    st.session_state.setdefault("chat_scroll_token", 0)
    st.session_state.setdefault(LLM_CACHE_KEY, {})

    if not st.session_state["selected_countries"]:
        st.session_state["selected_countries"] = (
            country_options[:MAX_COUNTRY_SELECTION] or DEFAULT_COUNTRY_SELECTION.copy()
        )
    if (
        not st.session_state["selected_products_display_specific"]
        and display_options
    ):
        st.session_state["selected_products_display_specific"] = display_options[
            : min(MAX_PRODUCT_SELECTION, len(display_options))
        ]


def reset_app_state(extra_widget_keys: Sequence[str] | None = None) -> None:
    st.session_state.messages = []
    st.session_state[LAST_RESULT_KEY] = None
    widget_keys = [
        "selected_countries",
        "selected_products_display_specific",
        "selected_products_display_categories",
        "product_mode",
    ]
    if extra_widget_keys:
        widget_keys.extend(extra_widget_keys)
    for widget_key in widget_keys:
        st.session_state.pop(widget_key, None)
    st.session_state["selected_product_codes"] = []
    st.session_state["correlation_signature"] = None
    st.session_state["analysis_active_run"] = None
    st.session_state["analysis_inflight"] = False
    st.session_state["analysis_request"] = None
    st.session_state["last_analysis_context"] = None
    st.session_state["chat_scroll_token"] = 0
    # llm_cache intentionally NOT cleared — cached results survive Clear Chat


def compute_selection_signature(countries: Sequence[str], products: Sequence[str]) -> tuple[
    tuple[str, ...], tuple[str, ...]
] | None:
    if not countries or not products:
        return None
    return (tuple(countries), tuple(products))


def enforce_selection_limit(key: str, max_items: int) -> tuple[list[str], bool]:
    """Trim a list in session_state before widgets using the same key are rendered."""
    selections = st.session_state.get(key, []) or []
    if not isinstance(selections, list):
        selections = list(selections)
        st.session_state[key] = selections
    trimmed = len(selections) > max_items
    if trimmed:
        st.session_state[key] = selections[:max_items]
        selections = st.session_state[key]
    return selections, trimmed
