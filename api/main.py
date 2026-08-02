import asyncio
import json
import logging
import tempfile
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()

from ingestion import SUPPORTED_UPLOAD_EXTENSIONS, ingest_documents
from rag.constants import (
    DIRECT_ANSWER,
    GENERATE_ANSWER,
    RERANK_DOCUMENTS,
    RETRIEVE,
    REWRITE_QUERY,
    WEB_SEARCH,
)
from rag.language import _get_detector
from rag.reranker import _get_reranker
from retrieval.vectorstore import get_vectorstore

from api.document_registry import DocumentRegistry
from api.models.schemas import QuestionRequest, SourceReference

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
warnings.filterwarnings(
    "ignore", message="Pydantic serializer warnings", category=UserWarning
)

logger = logging.getLogger(__name__)
DOCUMENTS_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "documents.db"

document_registry = DocumentRegistry(str(DOCUMENTS_DB_PATH))

# Human-readable (Bosnian) labels for each CRAG stage, shown live in the UI.
STAGE_LABELS = {
    REWRITE_QUERY: "Preformulišem pitanje…",
    RETRIEVE: "Pretražujem bazu znanja…",
    RERANK_DOCUMENTS: "Rangiram rezultate po relevantnosti…",
    WEB_SEARCH: "Tražim dodatne informacije na webu…",
    GENERATE_ANSWER: "Generišem odgovor…",
    DIRECT_ANSWER: "Generišem odgovor…",
}
TERMINAL_NODES = {GENERATE_ANSWER, DIRECT_ANSWER}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load heavy models on startup so the first request is fast."""
    logger.info("Loading models...")
    _get_reranker()
    _get_detector()

    from rag.graph import app as rag_app

    app.state.rag = rag_app
    logger.info("Models loaded — ready to serve requests.")
    yield


app = FastAPI(
    title="AI Assistant API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


def _sse(event: str, payload: dict[str, Any]) -> str:
    """Format a single Server-Sent Events message."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_chat_events(
    question: str, chat_history: list[dict[str, str]]
) -> AsyncIterator[str]:
    """Run the RAG graph and yield SSE messages: live stage progress, then answer tokens."""
    sources: list[dict[str, Any]] = []
    in_generation_node = False

    try:
        async for event in app.state.rag.astream_events(
            {"question": question, "chat_history": chat_history}, version="v2"
        ):
            kind = event["event"]
            name = event.get("name")

            if kind == "on_chain_start" and name in STAGE_LABELS:
                yield _sse(
                    "stage",
                    {"node": name, "label": STAGE_LABELS[name], "status": "started"},
                )
                if name in TERMINAL_NODES:
                    in_generation_node = True

            elif kind == "on_chain_end" and name in STAGE_LABELS:
                yield _sse(
                    "stage",
                    {"node": name, "label": STAGE_LABELS[name], "status": "done"},
                )
                if name in TERMINAL_NODES:
                    in_generation_node = False
                    output = event["data"].get("output") or {}
                    raw_sources = output.get("sources") or []
                    sources = [
                        SourceReference(**src).model_dump() for src in raw_sources
                    ]

            # Only forward LLM token deltas produced by the answer-generation node
            # itself — routing/rewriting sub-calls also stream but aren't user-facing.
            elif kind == "on_chat_model_stream" and in_generation_node:
                chunk = event["data"].get("chunk")
                text = getattr(chunk, "content", "") if chunk else ""
                if text:
                    yield _sse("token", {"text": text})

        yield _sse("sources", {"sources": sources})
        yield _sse("done", {})
    except Exception as exc:
        logger.exception("Streaming chat failed")
        yield _sse("error", {"message": str(exc)})


@app.post("/chat")
async def chat(body: QuestionRequest):
    """Run the RAG workflow, streaming live stage progress and answer tokens over SSE."""
    logger.info("Received question: %s", body.question)

    chat_history = [
        {"role": msg.role, "content": msg.content} for msg in body.history
    ]

    return StreamingResponse(
        _stream_chat_events(body.question, chat_history),
        media_type="text/event-stream",
    )


@app.get("/documents")
async def list_documents():
    """List documents currently ingested into the knowledge base."""
    return {"documents": document_registry.list_documents()}


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Remove a document's chunks from the vector store and delete its registry entry."""
    document = document_registry.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    if document["vector_ids"]:
        await asyncio.to_thread(get_vectorstore().delete, ids=document["vector_ids"])

    document_registry.delete_document(document_id)

    return {"message": "Document deleted", "id": document_id}


@app.post("/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """Ingest uploaded documents into the knowledge base.

    Files are only written to a temporary directory for the duration of
    ingestion — once chunked and upserted into Pinecone, the raw upload is no
    longer needed, so nothing is kept in data/uploads.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    with tempfile.TemporaryDirectory(prefix="upload_") as tmp_dir:
        saved_paths: list[Path] = []
        original_filenames: dict[str, str] = {}
        for index, upload in enumerate(files):
            if not upload.filename:
                continue

            original_name = Path(upload.filename).name
            suffix = Path(original_name).suffix.lower()
            if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {suffix or 'unknown'}",
                )

            # A per-file subdirectory keeps duplicate filenames in one batch from colliding.
            file_dir = Path(tmp_dir) / str(index)
            file_dir.mkdir()
            destination = file_dir / original_name
            contents = await upload.read()
            destination.write_bytes(contents)
            saved_paths.append(destination)
            original_filenames[str(destination)] = original_name

        ingested = await asyncio.to_thread(
            ingest_documents, files=saved_paths, original_filenames=original_filenames
        )

        for entry in ingested:
            document_registry.add_document(
                original_filename=original_filenames.get(entry["file_path"], entry["filename"]),
                stored_filename=entry["filename"],
                file_type=entry["file_type"],
                chunk_count=entry["chunk_count"],
                vector_ids=entry["vector_ids"],
            )

        chunk_count = sum(entry["chunk_count"] for entry in ingested)

        return {
            "message": "Documents ingested successfully",
            "files": [path.name for path in saved_paths],
            "chunks": chunk_count,
        }
