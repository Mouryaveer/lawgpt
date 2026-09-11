"""Authority-aware evidence selection and structured context construction."""

from __future__ import annotations


AUTHORITY_WEIGHT = {
    "constitution": 5,
    "statute": 4,
    "act": 4,
    "supreme_court": 4,
    "case_supreme_court": 4,
    "high_court": 3,
    "case": 2,
    "secondary": 1,
}


def authority_weight(metadata: dict) -> int:
    kind = str(metadata.get("authority_level") or metadata.get("document_type") or "secondary").lower().replace(" ", "_")
    court = str(metadata.get("court", "")).lower()
    if "supreme" in court:
        kind = "supreme_court"
    elif "high court" in court:
        kind = "high_court"
    return AUTHORITY_WEIGHT.get(kind, 1)


def build_evidence_pack(candidates: list[dict], limit: int = 10) -> list[dict]:
    ordered = sorted(candidates, key=lambda item: (float(item.get("rerank_score", 0.0)), authority_weight(item.get("metadata", {})), float(item.get("fusion_score", 0.0))), reverse=True)
    evidence = []
    for index, candidate in enumerate(ordered[:limit], start=1):
        metadata = dict(candidate.get("metadata", {}))
        evidence.append({
            "evidence_id": f"EV{index:03d}",
            "document_id": candidate.get("id"),
            "document_type": metadata.get("document_type", "unknown"),
            "authority_level": metadata.get("authority_level", metadata.get("document_type", "unknown")),
            "title": metadata.get("act_title") or metadata.get("case_title") or metadata.get("title") or metadata.get("document_id") or candidate.get("id"),
            "section": metadata.get("section_number") or metadata.get("article_number") or metadata.get("section_title", ""),
            "text": candidate.get("text", ""),
            "source_url": metadata.get("source_url") or metadata.get("act_url") or metadata.get("pdf_url", ""),
            "score": candidate.get("rerank_score", candidate.get("fusion_score", 0.0)),
            "metadata": metadata,
        })
    return evidence
