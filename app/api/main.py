import os
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.core import memory
from contextlib import asynccontextmanager

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.generation.chain import answer_question, astream_answer
from app.ingestion.pipeline import ingest_pdf
import json
from fastapi.staticfiles import StaticFiles
from app.retrieval.retriever import reset_index


setup_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    os.makedirs(
        os.path.dirname(settings.sqlite_db_path) or ".",
        exist_ok=True,
    )

    memory.init_db()

    yield

app = FastAPI(title="Multimodal PDF RAG API", lifespan=lifespan)

_JOBS: dict[str, dict] = {}


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


def _resolve_session(session_id: str | None) -> str:
    """Returns a valid session id, creating one if none was given or the
    given one doesn't exist (e.g. stale browser localStorage)."""
    if session_id and memory.session_exists(session_id):
        return session_id
    return memory.create_session()


@app.post("/upload")
async def upload_pdf(file: UploadFile):
    """Uploads a single PDF, replacing whatever was previously active:
    the old file is deleted from disk and the entire index (Chroma +
    docstore) is wiped before the new one is ingested. Runs synchronously
    — the response only comes back once ingestion is fully done, so this
    call can take a while on larger PDFs."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    settings = get_settings()
    os.makedirs(settings.upload_dir, exist_ok=True)
    

    dest_path = os.path.join(settings.upload_dir, file.filename)
    with open(dest_path, "wb") as f:
        f.write(await file.read())

    try:
        result = await ingest_pdf(dest_path)
    except Exception as e:
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(500, f"Ingestion failed: {e}")

    return {
        "filename": file.filename,
        "status": "complete",
        "texts_indexed": result["texts_indexed"],
        "tables_indexed": result["tables_indexed"],
        "images_indexed": result["images_indexed"],
    }





#------------------conversational emory endlpoint----------------------

@app.get("/sessions")
async def list_session():
    return memory.list_sessions()

@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    if not memory.session_exists(session_id):
        raise HTTPException(404, "Unknown session_id")
    return [
        {"role": m.role, "content": m.content, "sources": m.sources, "created_at": m.created_at}
        for m in memory.get_messages(session_id)
    ]


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    memory.delete_session(session_id)
    return {"status": "deleted"}



#--------chat-------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    """Non-streaming chat — returns the full answer, sources, and the
    session_id (new or echoed back) so the caller can keep the thread
    going."""
    session_id = _resolve_session(req.session_id)
    settings = get_settings()
    try:
        history = memory.get_messages(session_id, limit=settings.memory_turns * 2)
        memory.add_message(session_id, "user", req.question)

        result = answer_question(req.question, history=history)

        memory.add_message(session_id, "assistant", result["answer"], sources=result["sources"])
        return {**result, "session_id": session_id}
    except Exception as e:
        logger.exception("Chat failed")
        raise HTTPException(500, str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Streaming chat via Server-Sent Events. Emits `chunk` events as
    text arrives, then one `done` event carrying the session_id and
    sources, then persists the full turn to SQLite."""
    session_id = _resolve_session(req.session_id)
    settings = get_settings()
    history = memory.get_messages(session_id, limit=settings.memory_turns * 2)
    memory.add_message(session_id, "user", req.question)

    async def event_gen():
        yield f"event: session\ndata: {json.dumps({'session_id': session_id})}\n\n"
        try:
            async for event in astream_answer(req.question, history=history):
                if event["type"] == "chunk":
                    yield f"event: chunk\ndata: {json.dumps({'text': event['text']})}\n\n"
                else:
                    memory.add_message(
                        session_id, "assistant", event["answer"], sources=event["sources"]
                    )
                    yield f"event: done\ndata: {json.dumps({'sources': event['sources'], 'session_id': session_id})}\n\n"
        except Exception as e:
            logger.exception("Streaming chat failed")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}


# --- Static UI ---
# Serves the chat frontend at "/". Mounted last so it never shadows the
# API routes above (StaticFiles only handles paths it actually finds a
# file for; html=True serves index.html for "/").
_static_dir = os.path.join(os.path.dirname(__file__), "..", "..", "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")







