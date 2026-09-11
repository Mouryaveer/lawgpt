"""Deterministic legal query understanding used only for retrieval hints."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class LegalQuery:
    original: str
    entities: tuple[str, ...]
    document_types: tuple[str, ...]
    legal_domain: str
    jurisdiction: str = "India"
    temporal_intent: str = "current"

    def as_dict(self) -> dict:
        return asdict(self)


def analyze_query(query: str) -> LegalQuery:
    text = " ".join(query.strip().split())
    entities: list[str] = []
    patterns = [
        r"\b(?:Article|Art\.)\s+[0-9A-Za-z-]+(?:\s*\([^)]+\))?",
        r"\bSection\s+[0-9A-Za-z-]+(?:\s*\([^)]+\))?",
        r"\bOrder\s+[IVXLC]+\s+Rule\s+\d+[A-Za-z-]*",
    ]
    for pattern in patterns:
        entities.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    # Exact case names and statutory phrases are retained as query text, never invented.
    for phrase in re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,5}\s+(?:v\.?|vs\.?|versus)\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,5}", text):
        entities.append(phrase)
    entities = list(dict.fromkeys(e.strip() for e in entities))

    lowered = text.lower()
    types: list[str] = []
    if any(token in lowered for token in ("article", "constitution", "fundamental right", "directive principle")):
        types.append("constitution")
    if any(token in lowered for token in ("section", "act", "statute", "ipc", "bnss", "crpc", "cpc", "evidence")):
        types.append("act")
    if any(token in lowered for token in ("supreme court", "high court", "judgment", "judgement", "case law", "precedent")):
        types.append("case")
    if not types:
        types = ["act", "constitution", "case"]

    domain = "general"
    domain_terms = {
        "criminal_procedure": ("bail", "chargesheet", "charge sheet", "arrest", "remand"),
        "criminal_law": ("murder", "theft", "culpable", "offence", "offense", "ipc", "bnss"),
        "constitutional_law": ("article", "fundamental right", "constitution", "writ"),
        "civil_procedure": ("appeal", "limitation", "order vii", "injunction", "cpc"),
        "contract": ("contract", "agreement", "breach", "specific performance"),
    }
    for candidate, terms in domain_terms.items():
        if any(term in lowered for term in terms):
            domain = candidate
            break
    temporal = "historical" if re.search(r"\b(?:in|under)\s+(?:the\s+year\s+)?(?:19|20)\d{2}\b|historical|then applicable", lowered) else "current"
    return LegalQuery(text, tuple(entities), tuple(dict.fromkeys(types)), domain, "India", temporal)
