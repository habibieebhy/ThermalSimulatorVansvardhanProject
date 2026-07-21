"""Text, URL, unit, and identity normalization utilities."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit


TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}


def clean_text(value: str | None) -> str:
    return " ".join((value or "").replace("\u00a0", " ").split())


def canonicalize_url(url: str, base_url: str | None = None) -> str:
    absolute = urljoin(base_url or url, url)
    split = urlsplit(absolute)
    scheme = split.scheme.casefold() or "https"
    hostname = (split.hostname or "").casefold()
    port = f":{split.port}" if split.port and split.port not in {80, 443} else ""
    path = re.sub(r"/{2,}", "/", split.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(
        sorted(
            (key, value)
            for key, value in parse_qsl(split.query, keep_blank_values=False)
            if key.casefold() not in TRACKING_PARAMETERS
        )
    )
    return urlunsplit((scheme, f"{hostname}{port}", path, query, ""))


def normalized_name(value: str) -> str:
    lowered = value.casefold()
    lowered = re.sub(r"\b(?:mattress|matress|bed)\b", " ", lowered)
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def name_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalized_name(left), normalized_name(right)).ratio()


def length_to_mm(value: float, unit: str) -> float:
    conversion = {
        "mm": 1.0,
        "millimeter": 1.0,
        "millimeters": 1.0,
        "cm": 10.0,
        "centimeter": 10.0,
        "centimeters": 10.0,
        "in": 25.4,
        "inch": 25.4,
        "inches": 25.4,
        '"': 25.4,
    }
    try:
        return float(value) * conversion[unit.casefold()]
    except KeyError as exc:
        raise ValueError(f"Unsupported length unit: {unit}") from exc


def parse_first_thickness_mm(text: str) -> float | None:
    patterns = (
        r"(?:total\s+)?thickness\s*(?:is|:|-)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|inches|inch|in|\")",
        r"(?:height|depth)\s*(?:is|:|-)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|inches|inch|in|\")",
        r"(\d+(?:\.\d+)?)\s*(mm|cm|inches|inch|in|\")\s+(?:thick\s+)?mattress\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return round(length_to_mm(float(match.group(1)), match.group(2)), 2)
    return None


def parse_density_kg_m3(text: str) -> float | None:
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:kg\s*/\s*m(?:\^?3|³)|kgm(?:\^?3|³)|kg\s+per\s+cubic\s+met(?:er|re))",
        text,
        flags=re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def firmness_score(value: str | None) -> float | None:
    if not value:
        return None
    normalized = normalized_name(value)
    mapping = {
        "very soft": 0.1,
        "plush": 0.2,
        "soft": 0.25,
        "medium soft": 0.4,
        "medium": 0.5,
        "medium firm": 0.65,
        "firm": 0.8,
        "extra firm": 0.95,
        "very firm": 0.95,
    }
    for label, score in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        if label in normalized:
            return score
    return None
