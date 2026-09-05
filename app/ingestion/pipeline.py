import asyncio
import time

from app.core.logging import get_logger
from app.ingestion.parser import parse_pdf
from app.ingestion.summarizer import summarize_images, summarize_tables, summarize_texts
from app.retrieval.retriever import get_retriever, index_elements

logger = get_logger(__name__)


async def ingest_pdf(pdf_path: str) -> dict:
    """Run the full pipeline for a single PDF. Returns a small summary dict
    suitable for returning from a job-status endpoint."""
    start = time.time()
    logger.info("Starting ingestion: %s", pdf_path)

    doc = parse_pdf(pdf_path)

    text_summaries = summarize_texts(doc.texts)
    table_summaries = await summarize_tables(doc.tables_html)
    image_summaries = await summarize_images(doc.images_b64)

    retriever = get_retriever()
    index_elements(retriever, doc.texts, text_summaries, "text", pdf_path, doc.text_pages)
    index_elements(retriever, doc.tables_html, table_summaries, "table", pdf_path, doc.table_pages)
    index_elements(retriever, doc.images_b64, image_summaries, "image", pdf_path, doc.image_pages)

    elapsed = time.time() - start
    result = {
        "pdf_path": pdf_path,
        "texts_indexed": len(doc.texts),
        "tables_indexed": len(doc.tables_html),
        "images_indexed": len(doc.images_b64),
        "elapsed_seconds": round(elapsed, 2),
        "status": "complete",
    }
    logger.info("Finished ingestion: %s", result)
    return result
