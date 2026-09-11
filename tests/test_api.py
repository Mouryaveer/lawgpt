from fastapi.testclient import TestClient

import app


def test_health_reports_safe_architecture_without_secrets():
    response = TestClient(app.app).get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["generation_model"] == "openai/gpt-oss-120b"
    assert body["embedding_model"] == "Qwen/Qwen3-Embedding-8B"
    assert body["bm25_enabled"] is True
    assert "GROQ_API_KEY" in body["missing_configuration"]
    assert "gsk_" not in response.text.lower()
    assert "pcsk_" not in response.text.lower()


def test_query_does_not_fallback_to_llm_only(monkeypatch):
    def fail(_query):
        raise app.RAGError("configuration_error", "missing")
    monkeypatch.setattr(app.rag_service, "query", fail)
    response = TestClient(app.app).post("/api/query", json={"query": "What does Article 21 protect?"})
    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]
