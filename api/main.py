import asyncio
import logging
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from ingestion import SUPPORTED_UPLOAD_EXTENSIONS, ingest_documents
from rag.language import _get_detector
from rag.reranker import _get_reranker

from api.models.schemas import AnswerResponse, ChatMessage, QuestionRequest, SourceReference

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
warnings.filterwarnings(
    "ignore", message="Pydantic serializer warnings", category=UserWarning
)

logger = logging.getLogger(__name__)
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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


@app.post("/chat", response_model=AnswerResponse)
async def chat(body: QuestionRequest):
    """Run the RAG workflow and return the generated answer with sources."""
    logger.info("Received question: %s", body.question)

    chat_history = [
        {"role": msg.role, "content": msg.content} for msg in body.history
    ]

    result = await asyncio.to_thread(
        app.state.rag.invoke,
        {"question": body.question, "chat_history": chat_history},
    )

    sources = [SourceReference(**src) for src in result.get("sources") or []]

    return AnswerResponse(answer=result["generation"], sources=sources)


@app.post("/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """Save uploaded documents and ingest them into the internal knowledge base."""
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    saved_paths: list[Path] = []
    for upload in files:
        if not upload.filename:
            continue

        suffix = Path(upload.filename).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {suffix or 'unknown'}",
            )

        destination = UPLOAD_DIR / f"{Path(upload.filename).stem}-{uuid4().hex}{suffix}"
        contents = await upload.read()
        destination.write_bytes(contents)
        saved_paths.append(destination)

    chunk_count = await asyncio.to_thread(ingest_documents, files=saved_paths)

    return {
        "message": "Documents ingested successfully",
        "files": [path.name for path in saved_paths],
        "chunks": chunk_count,
    }
