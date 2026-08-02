import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
from fastapi.testclient import TestClient

load_dotenv()

import api.main as main_module
from rag.constants import GENERATE_ANSWER, REWRITE_QUERY

client = TestClient(main_module.app)


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse raw SSE response text into (event_type, payload) tuples."""
    events: list[tuple[str, dict]] = []
    for block in text.strip().split("\n\n"):
        if not block:
            continue
        event_type, data_line = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_line = line[len("data:"):].strip()
        if event_type and data_line is not None:
            events.append((event_type, json.loads(data_line)))
    return events


def test_health() -> None:
    """The health endpoint reports OK without touching any dependencies."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch.object(main_module.document_registry, "list_documents")
def test_list_documents(mock_list: MagicMock) -> None:
    """Documents are returned as-is from the registry."""
    mock_list.return_value = [{"id": "1", "original_filename": "a.txt", "chunk_count": 2}]

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == {"documents": mock_list.return_value}


@patch.object(main_module.document_registry, "get_document", return_value=None)
def test_delete_document_not_found(_mock_get: MagicMock) -> None:
    """Deleting an unknown document id returns 404."""
    response = client.delete("/documents/missing-id")

    assert response.status_code == 404


@patch.object(main_module.document_registry, "delete_document")
@patch.object(main_module, "get_vectorstore")
@patch.object(
    main_module.document_registry,
    "get_document",
    return_value={"id": "doc-1", "vector_ids": ["v1", "v2"]},
)
def test_delete_document_removes_vectors_and_registry_entry(
    _mock_get: MagicMock, mock_get_vectorstore: MagicMock, mock_delete_document: MagicMock
) -> None:
    """Deleting a known document removes its vectors then its registry row."""
    mock_vectorstore = MagicMock()
    mock_get_vectorstore.return_value = mock_vectorstore

    response = client.delete("/documents/doc-1")

    assert response.status_code == 200
    mock_vectorstore.delete.assert_called_once_with(ids=["v1", "v2"])
    mock_delete_document.assert_called_once_with("doc-1")


def test_upload_rejects_unsupported_extension() -> None:
    """Uploading a file with an unsupported extension returns 400."""
    response = client.post(
        "/documents/upload",
        files=[("files", ("bad.xyz", b"content", "text/plain"))],
    )

    assert response.status_code == 400


@patch.object(main_module.document_registry, "add_document")
@patch.object(main_module, "ingest_documents")
def test_upload_ingests_supported_file(
    mock_ingest: MagicMock, _mock_add_document: MagicMock
) -> None:
    """A supported file is ingested and its chunk count reported back."""
    mock_ingest.return_value = [
        {
            "file_path": "irrelevant",
            "filename": "note-123.txt",
            "file_type": ".txt",
            "chunk_count": 3,
            "vector_ids": ["v1", "v2", "v3"],
        }
    ]

    response = client.post(
        "/documents/upload",
        files=[("files", ("note.txt", b"some content", "text/plain"))],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["chunks"] == 3
    assert body["files"] == ["note.txt"]


def test_chat_streams_only_generation_node_tokens() -> None:
    """Only tokens emitted while the terminal generation node is running reach the client."""

    async def fake_astream_events(_inputs, version="v2"):
        yield {"event": "on_chain_start", "name": REWRITE_QUERY, "data": {}}
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": SimpleNamespace(content="LEAK")},
        }
        yield {"event": "on_chain_end", "name": REWRITE_QUERY, "data": {}}
        yield {"event": "on_chain_start", "name": GENERATE_ANSWER, "data": {}}
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": SimpleNamespace(content="Zdravo")},
        }
        yield {
            "event": "on_chain_end",
            "name": GENERATE_ANSWER,
            "data": {"output": {"sources": [{"source": "a.pdf", "snippet": "..."}]}},
        }

    main_module.app.state.rag = SimpleNamespace(astream_events=fake_astream_events)

    response = client.post("/chat", json={"question": "Pitanje?", "history": []})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    event_types = [event_type for event_type, _ in events]

    assert "done" in event_types
    tokens = "".join(payload["text"] for event_type, payload in events if event_type == "token")
    assert tokens == "Zdravo"
    sources_events = [payload for event_type, payload in events if event_type == "sources"]
    assert sources_events[-1]["sources"][0]["source"] == "a.pdf"
