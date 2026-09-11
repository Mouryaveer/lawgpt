"""Compatibility entrypoint for the versioned legal ingestion pipeline.

The previous module duplicated retrieval and model configuration. Production
ingestion now lives in ``python -m data.ingest`` so it cannot drift from the
API runtime.
"""

from data.ingest import build_bm25, create_index, upload
from config import get_settings


def build_new_data_index():
    settings = get_settings()
    return build_bm25(settings)


if __name__ == "__main__":
    settings = get_settings()
    print("BM25 records:", build_bm25(settings))
