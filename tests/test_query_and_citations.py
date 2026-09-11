from citations.verify import verify_answer
from query.expansion import expand_query
from query.understanding import analyze_query


def test_query_understanding_extracts_legal_entities():
    result = analyze_query("Can regular bail under Section 439 be granted after chargesheet?")
    assert "Section 439" in result.entities
    assert "criminal_procedure" == result.legal_domain
    assert "act" in result.document_types


def test_expansion_is_bounded_and_deterministic():
    result = expand_query(analyze_query("What does Article 21 protect?"))
    assert result[0] == "What does Article 21 protect?"
    assert len(result) <= 4


def test_citation_verification_suppresses_unsupported_claims():
    evidence = [{"evidence_id": "EV001", "text": "Section 439 permits the High Court or Court of Session to grant bail.", "title": "Code", "document_type": "act"}]
    draft = {"answer": "Section 439 permits bail.", "claims": [{"claim": "Section 439 permits the court to grant bail.", "evidence_ids": ["EV001"]}, {"claim": "The penalty is imprisonment for life.", "evidence_ids": ["EV001"]}], "confidence": "high"}
    result = verify_answer(draft, evidence)
    assert len(result["claims"]) == 1
    assert result["unsupported_claim_count"] == 1

