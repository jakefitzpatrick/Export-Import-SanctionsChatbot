"""Shared helper utilities for ImportInsight AI."""
from __future__ import annotations

import hashlib
import json

LLM_CACHE_KEY = "llm_cache"
LAST_RESULT_KEY = "latest_hts_result"


def llm_cache_key(countries: list[str], products: list[str]) -> str:
    """Stable cache key for Analyse summaries based on country/product selections."""
    payload = json.dumps({"c": sorted(countries), "p": sorted(products)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
