from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval.retriever import ID_KEY, get_retriever
from app.core.memory import Message

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a PDF document "
    "using the retrieved context below (which may include text passages, "
    "tables, and images from the document). "
    "Answer ONLY using the provided context. If the context doesn't contain "
    "the answer, say so clearly instead of guessing. "
    "After your answer, add a line 'Sources:' listing the page numbers of "
    "the context you actually used."
)

def _get_model()->ChatGroq:
    settings = get_settings()
    return ChatGroq(
        model = settings.answer_model,
        api_key=settings.groq_api_key,
        max_tokens=500,
    )

def _resolve_originals(retriever, summary_docs: list[Document]) -> list[tuple[Document, str]]:
    """Given the summary Documents returned by vector search, pull the
    ORIGINAL content (text/table-html/image-b64) from the docstore."""
    ids = [d.metadata[ID_KEY] for d in summary_docs]
    originals = retriever.docstore.mget(ids)
    return list(zip(summary_docs, originals))


def _history_messages(history: list[Message] | None)->list:
    if not history:
        return []
     
    turns = []
    for m in history:
        content = m.content if isinstance(m.content, str) else str(m.content)
        if m.role == "user":
            turns.append(HumanMessage(content=content))
        else:
            turns.append(AIMessage(content=content))
    return turns
    

          


def _build_messages(question: str, resolved: list[tuple[Document, str]], history: list[Message] | None = None) -> list:
    parts: list[str] = [f"Question:{question}\n\nContext:"]
 
    for summary_doc, original in resolved:
        if original is None:
            continue
        kind = summary_doc.metadata.get('kind', 'text')
        page = summary_doc.metadata.get('page', 'unknown')
 
        if kind == "image":
            try:
                description = summary_doc.page_content
                if not description:
                    raise ValueError("empty summary for image")
                parts.append(f"\n[Image from page {page}] Description: {description}")
            except Exception as e:
                logger.warning(f"Skipping image doc on page {page}: {e}")
                parts.append(f"\n[Image from page {page}] (summary unavailable)")
        elif kind == "table":
            parts.append(f"\n[Table from page {page}]\n{original}")
        else:
            parts.append(f"\n[Text from page {page}]\n{original}")
 
    human_content = "\n".join(parts)
    return [SystemMessage(content=_SYSTEM_PROMPT), *_history_messages(history), HumanMessage(content=human_content)]



def _sources_from(resolved: list[tuple[Document, str]]) -> list[dict]:
    return [
        {"kind": d.metadata.get("kind"), "page": d.metadata.get("page"), "source": d.metadata.get("source")}
        for d, original in resolved if original is not None
    ]


def answer_question(question: str, history: list[Message])-> dict:
    retriever = get_retriever()
    summary_docs = retriever.vectorstore.similarity_search(
        question
    )
    resolved = _resolve_originals(retriever, summary_docs)
    messages = _build_messages(question, resolved, history)

    model = _get_model()
    response = model.invoke(messages)

    sources = [
        {"kind": d.metadata.get("kind"), "page": d.metadata.get("page"), "source": d.metadata.get("source")}
        for d, original in resolved if original is not None
    ]
    return {"answer": response.content, "sources": sources}



async def astream_answer(question: str, history: list[Message]|None = None):
    """Streaming entry point for the FastAPI endpoint (Server-Sent Events).
    Yields text chunks as they're generated."""
    retriever = get_retriever()
    summary_docs = retriever.vectorstore.similarity_search(question)
    resolved = _resolve_originals(retriever, summary_docs)
    messages = _build_messages(question, resolved, history)

    model = _get_model()
    full_text=[]
    async for chunk in model.astream(messages):
        if chunk.content:
            full_text.append(chunk.content)
            yield {"type": "chunk", "text": chunk.content}
    yield {"type": "done", "answer": "".join(full_text), "sources": _sources_from(resolved)}