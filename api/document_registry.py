import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentRegistry:
    """SQLite-backed registry of documents ingested into the vector store.

    Tracks the Pinecone vector IDs produced for each ingested file so that a
    document can later be fully removed from the index (Pinecone serverless
    only supports delete-by-ID, not delete-by-metadata-filter).
    """

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id                TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    stored_filename   TEXT NOT NULL,
                    file_type         TEXT NOT NULL,
                    chunk_count       INTEGER NOT NULL,
                    vector_ids        TEXT NOT NULL,
                    uploaded_at       TEXT NOT NULL
                );
                """
            )

    def add_document(
        self,
        original_filename: str,
        stored_filename: str,
        file_type: str,
        chunk_count: int,
        vector_ids: list[str],
    ) -> dict:
        """Insert a registry row for a freshly ingested file and return it."""
        doc = {
            "id": str(uuid.uuid4()),
            "original_filename": original_filename,
            "stored_filename": stored_filename,
            "file_type": file_type,
            "chunk_count": chunk_count,
            "vector_ids": vector_ids,
            "uploaded_at": _now(),
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents
                (id, original_filename, stored_filename, file_type, chunk_count, vector_ids, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc["id"],
                    doc["original_filename"],
                    doc["stored_filename"],
                    doc["file_type"],
                    doc["chunk_count"],
                    json.dumps(vector_ids),
                    doc["uploaded_at"],
                ),
            )
        return doc

    def list_documents(self) -> list[dict]:
        """Return all registered documents, most recently uploaded first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, original_filename, file_type, chunk_count, uploaded_at
                FROM documents
                ORDER BY uploaded_at DESC, rowid DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_document(self, doc_id: str) -> dict | None:
        """Return a single document (including its vector IDs), or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        if row is None:
            return None
        doc = dict(row)
        doc["vector_ids"] = json.loads(doc["vector_ids"])
        return doc

    def delete_document(self, doc_id: str) -> None:
        """Remove a document's registry row (does not touch the vector store)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
