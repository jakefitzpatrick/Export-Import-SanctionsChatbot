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
    """Parse a special-duty string like 'Free (A+,AU)' into structured metadata."""
    if value is None:
        return None
    raw_text = str(value).strip()
    if not raw_text:
        return None

    rate_text = raw_text
    program_codes: list[str] = []
    match = PROGRAM_TOKEN_PATTERN.search(raw_text)
    if match:
        rate_text = raw_text[: match.start()].strip(" ,;")
        program_codes = [
            token.strip().upper()
            for token in match.group(1).split(",")
            if token.strip()
        ]

    if not program_codes:
        return None

    program_map = load_special_program_map()
    resolved_countries: dict[str, list[str]] = {}
    dynamic_codes: list[str] = []
    unknown_codes: list[str] = []
    labels: dict[str, str] = {}

    for code in program_codes:
        meta = program_map.get(code)
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
