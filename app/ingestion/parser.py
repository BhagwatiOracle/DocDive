from dataclasses import dataclass, field
from unstructured.partition.pdf import partition_pdf
from app.core.config import get_settings
from pathlib import Path
from app.core.logging import get_logger
import os
import base64
import shutil

logger = get_logger(__name__)

@dataclass
class ParsedDocument:
    texts: list[str] = field(default_factory=list)
    tables_html: list[str] = field(default_factory=list)
    images_b64: list[str] = field(default_factory=list)

    text_pages: list[int] = field(default_factory=list)
    table_pages: list[int] = field(default_factory=list)
    image_pages: list[int]=field(default_factory=list)


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_pdf(pdf_path: str, image_output_dir:str|None = None)->ParsedDocument:

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    settings = get_settings()
    image_output_dir = settings.extract_image

    if os.path.exists(image_output_dir):
        shutil.rmtree(image_output_dir)
    os.makedirs(image_output_dir, exist_ok=True)

    logger.info("Parsing PDF: %s", pdf_path)
    try:
        elements = partition_pdf(
            filename=pdf_path,
            strategy=settings.pdf_parse_strategy,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=settings.chunk_max_chars,
            combine_text_under_n_chars=settings.chunk_combine_under_chars,
            extract_image_block_output_dir=image_output_dir,
        )
    except Exception:
        logger.exception("Failed to parse PDF %s", pdf_path)
        raise

    doc = ParsedDocument()
    for el in elements:
        page_number = getattr(el.metadata, "page_number", None) or 0
        category = type(el).__name__

        if category == "Table":
            html = getattr(el.metadata, "text_as_html", None) or str(el)
            doc.tables_html.append(html)
            doc.table_pages.append(page_number)
        else:
            text = str(el).strip()
            if text:
                doc.texts.append(text)
                doc.text_pages.append(page_number)

    
    if os.path.isdir(image_output_dir):
        for fname in sorted(os.listdir(image_output_dir)):
            if fname.lower().endswith(('.png','.jpg','.jpeg')):
                fpath =os.path.join(image_output_dir, fname)
                doc.images_b64.append(_encode_image(fpath))
                doc.image_pages.append(0)
    

    logger.info(
        "Parsed %d text chunks, %d tables, %d images from %s",
        len(doc.texts), len(doc.tables_html), len(doc.images_b64), pdf_path,
    )
    
    return doc

