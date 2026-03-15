import logging
import warnings
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

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

    result = app.state.rag.invoke(
        {"question": body.question, "chat_history": chat_history}
    )

    sources = [
        SourceReference(**src) for src in result.get("sources") or []
    ]

    return AnswerResponse(answer=result["generation"], sources=sources)
