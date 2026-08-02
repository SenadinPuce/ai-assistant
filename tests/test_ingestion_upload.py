import os
from pathlib import Path

import pytest
from langchain_core.documents import Document

os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")

from ingestion import _group_ids_by_source_file, extract_text_from_file


@pytest.mark.parametrize(
    ("suffix", "content"),
    [
        (".txt", "Prvi red\nDrugi red"),
        (".md", "# Naslov\n\nTekst"),
    ],
)
def test_extract_text_from_plain_text_files(tmp_path: Path, suffix: str, content: str) -> None:
    file_path = tmp_path / f"sample{suffix}"
    file_path.write_text(content, encoding="utf-8")

    assert extract_text_from_file(file_path) == content


def test_group_ids_by_source_file_groups_chunks_per_file() -> None:
    """_group_ids_by_source_file groups vector IDs per originating file, in order."""
    doc_splits = [
        Document(page_content="A1", metadata={"file_path": "/tmp/a.txt", "file_type": ".txt"}),
        Document(page_content="A2", metadata={"file_path": "/tmp/a.txt", "file_type": ".txt"}),
        Document(page_content="B1", metadata={"file_path": "/tmp/b.pdf", "file_type": ".pdf"}),
    ]
    vector_ids = ["id-1", "id-2", "id-3"]

    grouped = _group_ids_by_source_file(doc_splits, vector_ids)

    assert grouped == [
        {
            "file_path": "/tmp/a.txt",
            "filename": "a.txt",
            "file_type": ".txt",
            "chunk_count": 2,
            "vector_ids": ["id-1", "id-2"],
        },
        {
            "file_path": "/tmp/b.pdf",
            "filename": "b.pdf",
            "file_type": ".pdf",
            "chunk_count": 1,
            "vector_ids": ["id-3"],
        },
    ]

