from retrieval.hybrid import HybridRetriever


class FakeEmbeddings:
    def embed_query(self, query):
        return [1.0]


class FakePinecone:
    def search(self, vector, limit, metadata_filter=None):
        return [{"id": "semantic", "text": "semantic", "metadata": {}, "score": 0.9}]


class FakeBM25:
    def search(self, query, limit):
        return [{"id": "lexical", "text": "lexical", "metadata": {}, "score": 1.0}]


def test_hybrid_fusion_merges_and_deduplicates():
    result = HybridRetriever(FakeEmbeddings(), FakePinecone(), FakeBM25(), 10).retrieve(["Section 302"], ("act",))
    assert {item.id for item in result} == {"semantic", "lexical"}
    assert all(item.fusion_score > 0 for item in result)

