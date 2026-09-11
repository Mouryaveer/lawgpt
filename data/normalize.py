"""Canonical metadata normalization shared by BM25 and Pinecone ingestion."""

from __future__ import annotations


def normalize_record(row: dict, index: int = 0) -> dict:
    metadata = dict(row.get("metadata") or {})
    record_id = str(row.get("id") or metadata.get("id") or f"chunk-{index}")
    text = str(row.get("text") or metadata.get("text") or "").strip()
    document_type = str(metadata.get("document_type") or metadata.get("source_type", "")).lower()
    if "constitution" in document_type or metadata.get("article_number") or metadata.get("part"):
        document_type = "constitution"
    elif "case" in document_type or metadata.get("case_title") or metadata.get("court"):
        document_type = "case"
    else:
        document_type = "act"
    authority = "constitution" if document_type == "constitution" else "statute" if document_type == "act" else "case"
    court = str(metadata.get("court", ""))
    if "supreme" in court.lower():
        authority = "supreme_court"
    elif "high court" in court.lower():
        authority = "high_court"
    canonical = {
        "document_id": str(metadata.get("document_id") or metadata.get("act_id") or metadata.get("case_id") or record_id),
        "parent_id": str(metadata.get("parent_id") or metadata.get("act_id") or metadata.get("case_id") or record_id),
        "document_type": document_type,
        "authority_level": authority,
        "act_title": metadata.get("act_title", ""),
        "section_number": metadata.get("section_number", ""),
        "section_title": metadata.get("section_title", ""),
        "article_number": metadata.get("article_number", ""),
        "case_title": metadata.get("case_title", ""),
        "case_year": metadata.get("case_year", metadata.get("year", "")),
        "court": court,
        "jurisdiction": metadata.get("jurisdiction", "India"),
        "legal_domain": metadata.get("legal_domain", "general"),
        "chunk_index": metadata.get("chunk_index", index),
        "total_chunks": metadata.get("total_chunks", metadata.get("total_chunks_in_document", 1)),
        "source_url": metadata.get("source_url", metadata.get("act_url", metadata.get("pdf_url", ""))),
        "pdf_url": metadata.get("pdf_url", ""),
        "effective_from": metadata.get("effective_from", ""),
        "effective_until": metadata.get("effective_until", ""),
        "status": metadata.get("status", "current"),
        "version": metadata.get("version", ""),
    }
    return {"id": record_id, "text": text, "metadata": canonical}
