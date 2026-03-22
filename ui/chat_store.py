import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatStore:
    """SQLite-backed store for chat sessions and their messages."""

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id         TEXT PRIMARY KEY,
                    title      TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id    TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        return dict(row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_chats(self) -> list[dict]:
        """Return all chats ordered by most recently updated first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM chats ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def create_chat(self) -> dict:
        """Insert a new empty chat and return it."""
        chat = {
            "id": str(uuid.uuid4()),
            "title": "Novi razgovor",
            "created_at": _now(),
            "updated_at": _now(),
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (chat["id"], chat["title"], chat["created_at"], chat["updated_at"]),
            )
        return chat

    def get_chat(self, chat_id: str) -> dict | None:
        """Return a single chat by ID, or None if it doesn't exist."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM chats WHERE id = ?",
                (chat_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_messages(self, chat_id: str) -> list[dict]:
        """Return all messages for a chat in chronological order."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id ASC",
                (chat_id,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def add_message(self, chat_id: str, role: str, content: str) -> None:
        """Append a message and bump the chat's updated_at timestamp."""
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (chat_id, role, content, now),
            )
            conn.execute(
                "UPDATE chats SET updated_at = ? WHERE id = ?",
                (now, chat_id),
            )

    def rename_chat(self, chat_id: str, title: str) -> None:
        """Update the title of a chat."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                (title.strip(), _now(), chat_id),
            )

    def delete_chat(self, chat_id: str) -> None:
        """Delete a chat and all its messages (cascades via FK)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
