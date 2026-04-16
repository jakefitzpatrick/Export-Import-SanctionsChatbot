"""Duty-rate parsing helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

FREE_DUTY_VALUES = {"free", "", "n/a", "none", "no", "zero", "0", "0.0", "0%", "0.0%"}
PERCENT_PATTERN = re.compile(r"(?P<value>-?\d+(?:\.\d+)?)\s*%")
CURRENCY_PREFIX_PATTERN = re.compile(
    r"(?P<symbol>[$])\s*(?P<value>-?\d+(?:\.\d+)?)(?P<unit>(?:\s*/\s*[A-Za-z0-9().-]+|\s+per\s+[A-Za-z0-9().-]+|\s+[A-Za-z0-9().-]+)?)",
    re.IGNORECASE,
)
CURRENCY_SUFFIX_PATTERN = re.compile(
    r"(?P<value>-?\d+(?:\.\d+)?)(?P<symbol>¢|cents?|dollars?)(?P<unit>(?:\s*/\s*[A-Za-z0-9().-]+|\s+per\s+[A-Za-z0-9().-]+|\s+[A-Za-z0-9().-]+)?)",
    re.IGNORECASE,
)
PER_PATTERN = re.compile(
    r"(?P<value>-?\d+(?:\.\d+)?)(?:\s*)(?:/|per)\s*(?P<unit>[A-Za-z0-9().-]+)",
    re.IGNORECASE,
)


@dataclass
class DutyComponent:
    """Atomic parsed component of a duty string."""

    type: str  # "ad_valorem" | "specific"
    value: float | None
    unit: str | None
    raw_segment: str


@dataclass
class DutyRate:
    raw_text: str
    kind: str
    ad_valorem_rate: float | None = None
    specific_amount: float | None = None
    specific_unit: str | None = None
    notes: str | None = None
    components: list[DutyComponent] = field(default_factory=list)

    @property
    def has_ad_valorem(self) -> bool:
        return any(comp.type == "ad_valorem" for comp in (self.components or []))

    @property
    def has_specific(self) -> bool:
        return any(comp.type == "specific" for comp in (self.components or []))


def _span_overlaps(span: tuple[int, int], spans: Iterable[tuple[int, int]]) -> bool:
    start, end = span
    return any(not (end <= s or start >= e) for s, e in spans)


def _clean_unit(fragment: str | None) -> str | None:
    if not fragment:
        return None
    cleaned = fragment.strip(" ,.;")
    return cleaned or None


def _gather_percent_components(text: str) -> tuple[list[DutyComponent], list[tuple[int, int]]]:
    components: list[DutyComponent] = []
    spans: list[tuple[int, int]] = []
    for match in PERCENT_PATTERN.finditer(text):
        value = float(match.group("value"))
        components.append(
            DutyComponent(
                type="ad_valorem",
                value=value,
                unit="percent",
                raw_segment=match.group(0).strip(),
            )
        )
        spans.append(match.span())
    return components, spans


def _gather_specific_components(text: str, consumed: list[tuple[int, int]]) -> list[DutyComponent]:
    components: list[DutyComponent] = []

    def _record(match: re.Match[str], raw: str) -> None:
        span = match.span()
        if _span_overlaps(span, consumed):
            return
        try:
            value = float(match.group("value"))
        except (TypeError, ValueError):
            return
        unit_fragment = match.groupdict().get("unit")
        components.append(
            DutyComponent(
                type="specific",
                value=value,
                unit=_clean_unit(unit_fragment),
                raw_segment=raw.strip(),
            )
        )
        consumed.append(span)

    for pattern in (CURRENCY_PREFIX_PATTERN, CURRENCY_SUFFIX_PATTERN, PER_PATTERN):
        for m in pattern.finditer(text):
            _record(m, m.group(0))

    return components


def parse_general_duty(value: str | float | int | None) -> DutyRate:
    if value is None:
        return DutyRate(raw_text="", kind="text", notes="missing duty", components=[])
    if isinstance(value, (int, float)):
        return DutyRate(
            raw_text=str(value),
            kind="ad_valorem",
            ad_valorem_rate=float(value),
            components=[DutyComponent(type="ad_valorem", value=float(value), unit="percent", raw_segment=str(value))],
        )

    raw_text = str(value).strip()
    normalized = raw_text.lower()
    if normalized in FREE_DUTY_VALUES:
        return DutyRate(
            raw_text=raw_text,
            kind="ad_valorem",
            ad_valorem_rate=0.0,
            notes="duty-free entry",
            components=[DutyComponent(type="ad_valorem", value=0.0, unit="percent", raw_segment=raw_text)],
        )

    percent_components, consumed_spans = _gather_percent_components(raw_text)
    specific_components = _gather_specific_components(raw_text, consumed_spans)
    components = percent_components + specific_components

    if not components:
        return DutyRate(raw_text=raw_text, kind="text", notes="unparsable duty text", components=[])

    has_percent = bool(percent_components)
    has_specific = bool(specific_components)

    if has_percent and has_specific:
        kind = "mixed"
    elif has_percent:
        kind = "ad_valorem"
    elif has_specific:
        kind = "specific"
    else:
        kind = "text"

    ad_valorem_rate = next((comp.value for comp in percent_components), None)
    specific_component = next((comp for comp in specific_components if comp.value is not None), None)

    return DutyRate(
        raw_text=raw_text,
        kind=kind,
        ad_valorem_rate=ad_valorem_rate,
        specific_amount=specific_component.value if specific_component else None,
        specific_unit=specific_component.unit if specific_component else None,
        notes=None,
        components=components,
    )


def limit_to_general_duty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict displayed HTS columns to general duty and earlier fields."""
    if "general_duty_rate" not in df.columns:
        return df
    columns = [
        col
        for col in [
            "hts_code",
            "chapter",
            "heading",
            "subheading",
            "statistical_suffix",
            "indent_level",
            "description",
            "full_description",
            "unit",
            "general_duty_rate",
        ]
        if col in df.columns
    ]
    if not columns:
        return df
    return df.loc[:, columns]
