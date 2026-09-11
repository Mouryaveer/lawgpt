"""Persistent BM25-style lexical retrieval using SQLite FTS5."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from data.normalize import normalize_record


class BM25Error(RuntimeError):
    pass


def _tokens(text: str) -> str:
    return " ".join(re.findall(r"[\w/-]+", text.lower()))


class BM25Index:
    def __init__(self, database_path: str):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("SELECT 1 FROM chunks_fts LIMIT 1")
        except sqlite3.OperationalError as exc:
            raise BM25Error(f"BM25 index is not built at {self.path}") from exc

    @classmethod
    def build(cls, corpus_path: str, database_path: str) -> int:
        destination = Path(database_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(destination)
        connection.executescript("DROP TABLE IF EXISTS chunks; DROP TABLE IF EXISTS chunks_fts; CREATE TABLE chunks (id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata TEXT NOT NULL); CREATE VIRTUAL TABLE chunks_fts USING fts5(id UNINDEXED, text, metadata, tokenize='unicode61');")
        count = 0
        with open(corpus_path, "r", encoding="utf-8") as source:
            for index, line in enumerate(source):
                if not line.strip():
                    continue
                record = normalize_record(json.loads(line), index)
                metadata = record["metadata"]
                text = record["text"]
                connection.execute("INSERT OR REPLACE INTO chunks VALUES (?, ?, ?)", (record["id"], text, json.dumps(metadata, ensure_ascii=False)))
                searchable = _tokens(text + " " + json.dumps(metadata, ensure_ascii=False))
                connection.execute("INSERT INTO chunks_fts VALUES (?, ?, ?)", (record["id"], searchable, json.dumps(metadata, ensure_ascii=False)))
                count += 1
                if count % 1000 == 0:
                    connection.commit()
        connection.commit()
        connection.close()
        return count

    def search(self, query: str, limit: int = 50) -> list[dict]:
        safe_query = _tokens(query)
        if not safe_query:
            return []
        try:
            rows = self.connection.execute("SELECT id, text, metadata, bm25(chunks_fts) AS raw_score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY raw_score LIMIT ?", (safe_query, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        if not rows:
            return []
        scores = [max(0.0, -float(row["raw_score"])) for row in rows]
        maximum = max(scores) or 1.0
        return [{"id": row["id"], "text": row["text"], "metadata": json.loads(row["metadata"]), "score": score / maximum} for row, score in zip(rows, scores)]

    def get(self, record_id: str) -> dict | None:
        row = self.connection.execute("SELECT id, text, metadata FROM chunks WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            return None
        return {"id": row["id"], "text": row["text"], "metadata": json.loads(row["metadata"])}
