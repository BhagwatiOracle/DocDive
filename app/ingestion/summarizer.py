import hashlib
import os
import json
import asyncio
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from app.core.config import get_settings
from app.core.logging import get_logger


logger = get_logger(__name__)


_TABLE_PROMPT = (
    "You are creating a search index entry for a table extracted from a "
    "document. Summarize what this table contains: its subject, the "
    "columns/rows it has, and any key figures. Be specific enough that "
    "someone could find this table by searching for its contents. "
    "Table (HTML):\n\n{table}"
)

_IMAGE_PROMPT = (
    "You are creating a search index entry for an image/chart/figure "
    "extracted from a document. Describe precisely what it shows: chart "
    "type, axes/labels, key numbers or trends, or — if it's a photo or "
    "diagram — its literal content. Be specific enough that someone could "
    "find this image by searching for its contents."
)

_CACHE_PATH = "./data/docstore/summary_cache.json"

def _cache_key(content: str)->str:
    """Turns content into unique id"""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def _load_cache()->dict:
    if os.path.exists(_CACHE_PATH):
        with open(_CACHE_PATH, 'r') as f:
            return json.load(f)
    return {}

def _save_cache(cache: dict)->None:
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    with open(_CACHE_PATH, 'w') as f:
        json.dump(cache, f)

def _get_model()->ChatGroq:
    settings = get_settings()
    return ChatGroq(
        model = settings.summary_model,
        api_key=settings.groq_api_key,
        max_tokens=500,
    )

async def _summarize_table(model: ChatGroq, table_html: str, cache: dict, sem: asyncio.Semaphore)->str:
    key = _cache_key(table_html)
    if key in cache:
        return cache[key]
    async with sem:
        msg = await model.ainvoke([HumanMessage(content=_TABLE_PROMPT.format(table=table_html))])
    summary = msg.content if isinstance(msg.content, str) else str(msg.content)
    cache[key] = summary
    return summary

async def _summarize_image(model: ChatGroq, image_b64: str, cache: dict, sem: asyncio.Semaphore)-> str:
    key = _cache_key(image_b64[:200])
    if key in cache:
        return cache[key]
    async with sem:
        msg = await model.ainvoke([
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": _IMAGE_PROMPT
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                }
            ])
        ])

        summary = msg.content if isinstance(msg.content, str) else str(msg.content)
        cache[key] = summary
        return summary


async def summarize_tables(tables_html: list[str]) -> list[str]:
    if not tables_html:
        return []
    settings = get_settings()
    model = _get_model()
    cache = _load_cache()
    sem = asyncio.Semaphore(settings.max_concurrent_summaries)
    logger.info("Summarizing %d tables", len(tables_html))
    results = await asyncio.gather(*[_summarize_table(model, t, cache, sem) for t in tables_html])
    _save_cache(cache)
    return list(results)


async def summarize_images(images_b64: list[str]) -> list[str]:
    if not images_b64:
        return []
    settings = get_settings()
    model = _get_model()
    cache = _load_cache()
    sem = asyncio.Semaphore(settings.max_concurrent_summaries)
    logger.info("Summarizing %d images", len(images_b64))
    results = await asyncio.gather(*[_summarize_image(model, i, cache, sem) for i in images_b64])
    _save_cache(cache)
    return list(results)



def summarize_texts(texts: list[str]) -> list[str]:
    """Text chunks are usually fine to embed directly (no LLM call needed —
    this keeps cost down since text is the majority of most PDFs). Return
    as-is; kept as a function so callers have one uniform interface and it's
    a single place to add e.g. text summarization for very long chunks."""
    return texts