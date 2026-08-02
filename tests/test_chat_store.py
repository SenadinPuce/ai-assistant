import sqlite3
from pathlib import Path

from ui.chat_store import ChatStore


def _make_store(tmp_path: Path) -> ChatStore:
    return ChatStore(str(tmp_path / "chats.db"))


def test_create_chat_has_default_title(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    chat = store.create_chat()

    assert chat["title"] == "Novi razgovor"
    assert store.get_chat(chat["id"]) is not None


def test_get_chat_returns_none_when_missing(tmp_path: Path) -> None:
    store = _make_store(tmp_path)

    assert store.get_chat("missing-id") is None


def test_list_chats_orders_by_most_recently_updated(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    first = store.create_chat()
    second = store.create_chat()

    # Bumping the first chat's updated_at should move it back to the front.
    store.add_message(first["id"], "user", "Zdravo")

    chats = store.list_chats()

    assert [c["id"] for c in chats] == [first["id"], second["id"]]


def test_add_message_with_sources_roundtrips_as_json(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    chat = store.create_chat()
    sources = [{"source": "a.pdf", "page": 1, "snippet": "..."}]

    store.add_message(chat["id"], "assistant", "Odgovor [1].", sources=sources)

    messages = store.get_messages(chat["id"])
    assert messages[0]["sources"] == sources


def test_add_message_without_sources_returns_empty_list(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    chat = store.create_chat()

    store.add_message(chat["id"], "user", "Pitanje")

    messages = store.get_messages(chat["id"])
    assert messages[0]["sources"] == []


def test_rename_chat_updates_title(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    chat = store.create_chat()

    store.rename_chat(chat["id"], "  Novi naslov  ")

    assert store.get_chat(chat["id"])["title"] == "Novi naslov"


def test_delete_chat_cascades_to_messages(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    chat = store.create_chat()
    store.add_message(chat["id"], "user", "Pitanje")

    store.delete_chat(chat["id"])

    assert store.get_chat(chat["id"]) is None
    with sqlite3.connect(store._db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat["id"],)
        ).fetchone()[0]
    assert count == 0
