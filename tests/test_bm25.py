import json

from retrieval.bm25 import BM25Index


def test_bm25_finds_exact_section(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join([
        json.dumps({"id": "a", "text": "Section 420 deals with cheating.", "metadata": {"document_type": "act"}}),
        json.dumps({"id": "b", "text": "Article 21 protects life and personal liberty.", "metadata": {"document_type": "constitution"}}),
    ]), encoding="utf-8")
    database = tmp_path / "bm25.sqlite3"
    assert BM25Index.build(str(corpus), str(database)) == 2
    index = BM25Index(str(database))
    result = index.search("Section 420", 5)
    assert result and result[0]["id"] == "a"

