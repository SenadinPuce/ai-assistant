from pathlib import Path

from api.document_registry import DocumentRegistry


def _make_registry(tmp_path: Path) -> DocumentRegistry:
    return DocumentRegistry(str(tmp_path / "documents.db"))


def test_add_and_list_documents_orders_most_recent_first(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)

    first = registry.add_document("a.txt", "a-123.txt", ".txt", 2, ["id-1", "id-2"])
    second = registry.add_document("b.pdf", "b-456.pdf", ".pdf", 1, ["id-3"])

    documents = registry.list_documents()

    assert [doc["id"] for doc in documents] == [second["id"], first["id"]]
    assert documents[0]["chunk_count"] == 1
    assert "vector_ids" not in documents[0]


def test_get_document_includes_vector_ids(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)
    created = registry.add_document("a.txt", "a-123.txt", ".txt", 2, ["id-1", "id-2"])

    fetched = registry.get_document(created["id"])

    assert fetched is not None
    assert fetched["vector_ids"] == ["id-1", "id-2"]


def test_get_document_returns_none_when_missing(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)

    assert registry.get_document("missing-id") is None


def test_delete_document_removes_entry(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)
    created = registry.add_document("a.txt", "a-123.txt", ".txt", 2, ["id-1", "id-2"])

    registry.delete_document(created["id"])

    assert registry.get_document(created["id"]) is None
    assert registry.list_documents() == []
