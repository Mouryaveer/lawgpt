"""Claim-level citation verification against the supplied evidence pack."""

from __future__ import annotations

import re

STOPWORDS = {"the", "and", "that", "this", "with", "from", "under", "where", "which", "into", "are", "for", "not", "may", "can", "has", "have", "been", "was", "were", "will", "such", "their", "there"}


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]{3,}", text.lower()) if term not in STOPWORDS}


def verify_answer(draft: dict, evidence: list[dict]) -> dict:
    by_id = {item["evidence_id"]: item for item in evidence}
    verified_claims = []
    unsupported = 0
    for raw in draft.get("claims", []):
        if not isinstance(raw, dict) or not isinstance(raw.get("claim"), str):
            continue
        claim = raw["claim"].strip()
        ids = [eid for eid in raw.get("evidence_ids", []) if eid in by_id]
        claim_terms = _terms(claim)
        support = set().union(*(_terms(by_id[eid]["text"]) for eid in ids)) if ids else set()
        overlap = len(claim_terms & support) / max(1, len(claim_terms))
        status = "supported" if ids and overlap >= 0.35 else "partially_supported" if ids and overlap >= 0.15 else "unsupported"
        if status == "unsupported":
            unsupported += 1
            continue
        verified_claims.append({"claim": claim, "supporting_evidence": ids, "status": status})
    answer = draft.get("answer", "").strip()
    insufficient = bool(draft.get("insufficient_evidence")) or not evidence or not verified_claims
    if unsupported:
        # A draft paragraph cannot safely be surgically edited without a second
        # model pass. Reconstruct from only verified claims instead.
        lines = [f"- {item['claim']} ({', '.join(item['supporting_evidence'])})" for item in verified_claims]
        answer = "## Verified findings\n\n" + "\n".join(lines) if lines else ""
        answer += "\n\n> **Evidence note:** Unsupported draft propositions were omitted because the retrieved authorities did not verify them."
    if not answer:
        answer = "The available retrieved authorities do not establish a sufficiently verified answer to this question."
        insufficient = True
    return {"answer": answer, "claims": verified_claims, "unsupported_claim_count": unsupported, "grounded": bool(verified_claims) and unsupported == 0, "insufficient_evidence": insufficient, "confidence": draft.get("confidence", "low")}
