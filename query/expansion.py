"""Bounded, deterministic query expansion. No model-generated facts are added."""

from __future__ import annotations

import re

from .understanding import LegalQuery


def expand_query(analysis: LegalQuery, limit: int = 4) -> list[str]:
    queries = [analysis.original]
    original = analysis.original
    lowered = original.lower()
    for entity in analysis.entities:
        if entity.lower().startswith("section "):
            queries.append(f"{entity} {original}")
        elif entity.lower().startswith("article "):
            queries.append(f"{entity} constitutional law {original}")
    if "chargesheet" in lowered:
        queries.append(original.replace("chargesheet", "charge sheet"))
    if "charge sheet" in lowered:
        queries.append(original.replace("charge sheet", "chargesheet"))
    return list(dict.fromkeys(q for q in queries if q.strip()))[:limit]
