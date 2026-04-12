## 2026-04-11 — Analyse Button & Streaming Summary

- Cached the V-Dem risk CSV behind `get_risk_df()` (`@st.cache_resource`) so Streamlit reads it once per process instead of on every widget tweak.
- Fixed the `itertools.product` name collision by importing the module and referencing `itertools.product`, so HTS row variables no longer stomp the iterator.
- Swapped the auto-fire correlation pipeline for an explicit **Analyse** button in the context bar; it disables while a run is active and queues the selected countries/products for processing.
- Rebuilt the analysis runner to stream Azure OpenAI tokens into a chat placeholder, then persist the finished message (with charts + risk pills) in history once the stream closes.

## 2026-04-11 — Sticky Chat Shell

- Replaced the old textarea+button form with `st.chat_input`, so Enter submits, Shift+Enter inserts a newline, and the placeholder stays short. The send button now lives inside the composer shell thanks to fresh CSS overrides.
- Locked the layout so the context bar is sticky at the top, the chat feed is the only scrollable surface, and the composer sticks to the bottom. The sidebar remains intact but no longer houses the retired gauge.
- Added a lightweight auto-scroll hook that jumps to the newest chat message after every new user or assistant entry, matching modern LLM chat UX.
- Hooked message appends through a helper that also bumps a scroll token, which we use to drive the scrolling script. Clearing chat resets the token for a clean slate.

## 2026-04-10 — Risk & Tariff Correlation UX

- Added cached helpers that normalize the in-memory country risk table, hydrate product dropdown options from SQLite, and parse general duty rate strings. These keep the Streamlit callbacks lightweight and prevent redundant round-trips.
- Sidebar now exposes multi-select controls for both countries and HTS products. The first chosen country still drives the gauge widget for quick visual context.
- Introduced a “Risk & Tariff Analysis” button that orchestrates the LLM SQL generation, executes the query, and materializes the Cartesian country×product dataset. Results persist in `st.session_state` so reruns don’t thrash the model.
- Built a Plotly scatter chart that plots risk score (x) versus general duty rate (y), coloring markers by country and using HTS codes as symbols. A synchronized dataframe view lets compliance analysts sort/filter the raw pairs.
- Added an LLM narrative generator that ingests the correlation payload, highlights outliers, and injects the summary into both the dashboard section and the chat transcript.
- Fixed a startup regression by defining the `get_risk_color` helper before the risk DataFrame is instantiated so Streamlit can boot without NameErrors.
- Resolved Streamlit caching crash by marking the SQLite connection parameter as an underscore-prefixed argument inside `load_product_options`, which prevents Streamlit from trying to hash the unpicklable connection handle.
- Updated the LLM summary call to use the default temperature supported by the Azure deployment (the previous 0.4 value triggered an `unsupported_value` 400 error).
- Moved the country/product selectors into the main column, simplified the sidebar gauge into a three-zone semi-circle, and wired the correlation pipeline to run automatically (with debounce, run IDs, and in-flight/pending guards) whenever selections change. Added a refresh nudge, race-safe state handling, and richer empty-state messaging in the correlation section.
- Added guardrails so each multiselect is capped at three choices; trying to exceed the limit keeps only the first three selections and surfaces a warning. This prevents overly dense charts and controls API usage.
- Replaced the hardcoded `COUNTRY_RISK` data with live values from `vdem_risk_scored.csv`, deriving colors/levels on the fly and falling back to the legacy defaults only when the CSV is missing. The sidebar gauge and correlation logic now stay in sync with the latest risk_model output.
- Consolidated all governance-risk utilities into `risk_model.py` (shared thresholds, color map, CSV loader) and removed the redundant `risk_config.py`/`governance.py` shims so every consumer uses the same source of truth.
- Rebuilt the Streamlit layout into a single-column chat experience: the country/product “Context Bar” sticks to the top, correlation analyses render as chart/table attachments inside assistant messages, and the composer sits sticky at the bottom like modern chat apps—no more sidebar gauge or standalone dashboard.
- Restored the left sidebar (logo, settings, clear-chat control, about) for quick orientation, while keeping risk insights embedded in the chat. Also replaced the LLM-generated HTS lookup query with a direct parameterized SQL call so multi-select changes no longer trigger invalid syntax or unnecessary API spend.
