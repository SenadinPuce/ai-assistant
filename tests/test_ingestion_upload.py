import os
from pathlib import Path

import pytest

os.environ.setdefault("PINECONE_INDEX_NAME", "test-index")

from ingestion import extract_text_from_file


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
