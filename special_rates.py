"""Helpers for parsing HTS special duty-rate programs."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from duty_parser import DutyRate, parse_general_duty
from logger import setup_logger

logger = setup_logger(__name__)

PROGRAM_TOKEN_PATTERN = re.compile(r"\(([^)]+)\)")

# Regex to capture "See xxxx.xx.xx (CODE)" secondary references.
# These appear after the main program list and grant the same free rate
# via a Chapter 98/99 provision for the parenthesised program code.
SEE_REFERENCE_PATTERN = re.compile(r"\bSee\s+[\d\w.,\-]+\s+\(([^)]+)\)", re.IGNORECASE)

# Codes that use asterisk/plus suffixes should fall back to their base
# program's country list when the variant itself has no static list.
# e.g. E* → CBI countries (same as E), S+ → USMCA countries (same as S).
_SUFFIX_FALLBACK: dict[str, str] = {
    "A*": "A",
    "E*": "E",
    "S+": "S",
    "P+": "P",
    "J+": "J",
    "J*": "J",
}
COUNTRY_ALIASES = {
    "burma": "Burma/Myanmar",
    "myanmar": "Burma/Myanmar",
    "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "gambia, the": "The Gambia",
    "the gambia": "The Gambia",
    "congo (kinshasa)": "Democratic Republic of the Congo",
    "democratic republic of congo": "Democratic Republic of the Congo",
    "congo (brazzaville)": "Republic of the Congo",
    "republic of congo": "Republic of the Congo",
    "timor-leste": "Timor-Leste",
}
_LOGGED_DYNAMIC_CODES: set[str] = set()
_LOGGED_UNKNOWN_CODES: set[str] = set()


def canonicalize_country(name: str) -> str:
    """Normalize country labels to match risk dataset naming."""
    if not name:
        return ""
    cleaned = name.strip()
    return COUNTRY_ALIASES.get(cleaned.lower(), cleaned)


@lru_cache(maxsize=1)
def load_special_program_map() -> dict[str, dict]:
    path = Path(__file__).resolve().parent / "data" / "special_program_codes.json"
    if not path.exists():
        logger.warning("Special program code map missing at %s", path)
        return {}
    with path.open() as handle:
        raw_map = json.load(handle)
    for code, meta in raw_map.items():
        countries = meta.get("countries") or []
        meta["countries"] = [canonicalize_country(country) for country in countries]
    return raw_map


@dataclass
class SpecialDutyRule:
    raw_text: str
    rate_text: str
    program_codes: list[str]
    duty_rate: DutyRate | None
    resolved_countries: dict[str, list[str]]
    dynamic_codes: list[str]
    unknown_codes: list[str]
    labels: dict[str, str]

    def applies_to(self, country: str) -> bool:
        canonical = canonicalize_country(country)
        return any(
            canonical in countries for countries in self.resolved_countries.values()
        )

    def codes_for_country(self, country: str) -> list[str]:
        canonical = canonicalize_country(country)
        return [
            code
            for code, countries in self.resolved_countries.items()
            if canonical in countries
        ]

    @property
    def has_dynamic_codes(self) -> bool:
        return bool(self.dynamic_codes)

    def format_labels(self, codes: Iterable[str]) -> str:
        labels = [self.labels.get(code, code) for code in codes]
        return ", ".join(labels)


def _record_dynamic_warning(code: str) -> None:
    if code not in _LOGGED_DYNAMIC_CODES:
        logger.warning("Special program %s recognized but not resolved", code)
        _LOGGED_DYNAMIC_CODES.add(code)


def _record_unknown_warning(code: str) -> None:
    if code not in _LOGGED_UNKNOWN_CODES:
        logger.warning("Unknown special program code %s in special duty rate", code)
        _LOGGED_UNKNOWN_CODES.add(code)


def parse_special_duty(value: str | None) -> SpecialDutyRule | None:
    """Parse a special-duty string like 'Free (A+,AU)' into structured metadata.

    Handles two syntactic forms found in HTS special-duty columns:

    1. Simple list: ``Free (A+,AU,BH,CL,...)``
       All codes are in the first parenthetical group.

    2. See-reference list: ``Free (BH,CL,...) See 9822.04.05 (AU) See 9823.01.01-9823.01.07 (S+)``
       The main list is in the first group; additional codes appear inside
       ``See xxxx.xx (CODE)`` fragments.  Both sets are merged so that, e.g.,
       Australia's free rate under a bilateral FTA is correctly recognised.

    Suffix-variant codes (``E*``, ``S+``, ``P+``, ``A*``, ``J*``, ``J+``) fall
    back to their base program's country list when the variant itself has no
    static country list in the program map.
    """
    if value is None:
        return None
    raw_text = str(value).strip()
    if not raw_text:
        return None

    rate_text = raw_text
    program_codes: list[str] = []

    # --- Step 1: extract codes from the FIRST parenthetical (main list) ---
    first_match = PROGRAM_TOKEN_PATTERN.search(raw_text)
    if first_match:
        rate_text = raw_text[: first_match.start()].strip(" ,;")
        program_codes = [
            token.strip().upper()
            for token in first_match.group(1).split(",")
            if token.strip()
        ]

    # --- Step 2: extract codes from any "See xxxx (CODE)" fragments ---
    for see_match in SEE_REFERENCE_PATTERN.finditer(raw_text):
        for token in see_match.group(1).split(","):
            code = token.strip().upper()
            if code and code not in program_codes:
                program_codes.append(code)

    if not program_codes:
        return None

    # --- Step 3: resolve each code against the program map ---
    program_map = load_special_program_map()
    resolved_countries: dict[str, list[str]] = {}
    dynamic_codes: list[str] = []
    unknown_codes: list[str] = []
    labels: dict[str, str] = {}

    for code in program_codes:
        meta = program_map.get(code)

        # Suffix-variant fallback: E* → E, S+ → S, P+ → P, etc.
        if (meta is None or meta.get("type") == "dynamic_program" or not meta.get("countries")):
            base_code = _SUFFIX_FALLBACK.get(code)
            if base_code:
                base_meta = program_map.get(base_code)
                if base_meta and base_meta.get("countries"):
                    # Use base program's countries; keep the variant's label if available
                    resolved_countries[code] = base_meta["countries"]
                    labels[code] = (meta or {}).get("label", base_meta.get("label", code))
                    continue

        if not meta:
            _record_unknown_warning(code)
            unknown_codes.append(code)
            continue

        labels[code] = meta.get("label", code)
        if meta.get("type") == "dynamic_program" or not meta.get("countries"):
            dynamic_codes.append(code)
            _record_dynamic_warning(code)
            continue

        resolved_countries[code] = meta["countries"]

    duty_rate = parse_general_duty(rate_text or raw_text)
    return SpecialDutyRule(
        raw_text=raw_text,
        rate_text=rate_text or raw_text,
        program_codes=program_codes,
        duty_rate=duty_rate,
        resolved_countries=resolved_countries,
        dynamic_codes=dynamic_codes,
        unknown_codes=unknown_codes,
        labels=labels,
    )


__all__ = [
    "SpecialDutyRule",
    "canonicalize_country",
    "load_special_program_map",
    "parse_special_duty",
]
