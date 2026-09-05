from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import get_settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from typing import Any
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_core.callbacks import CallbackManagerForRetrieverRun
import uuid
from app.core.logging import get_logger
from app.core.docstore import get_docstore

logger = get_logger(__name__)

ID_KEY = "doc_id"


def _get_vectorstore() -> Chroma:
    settings = get_settings()
    embeddings = HuggingFaceEmbeddings(
        model=settings.embedding_model
    )
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )

def _load_all_summary_document(vectorstore: Chroma)->list[Document]:
    raw = vectorstore.get(include=['documents','metadatas'])
    docs = [
        Document(page_content=text,metadata=meta or {} )
        for text, meta in zip(raw['documents'], raw['metadatas'])
    ]
    return docs

def _build_bm25_retriever(vectorstore: Chroma, k: int)-> BM25Retriever:
    docs = _load_all_summary_document(vectorstore)
    if not docs:
        return None
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k=k
    return bm25


def _build_reranker(top_n:int)->CohereRerank:
    settings = get_settings()
    return CohereRerank(model='rerank-english-v3.0', top_n=top_n, cohere_api_key=settings.cohere_api_key)



class HybridMultiVectorRetriever(BaseRetriever):

    # allows the class to have custom python object without pydantic complaning
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vectorstore: Chroma
    docstore: Any
    dense_k: int = 5
    sparse_k: int = 5
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rerank_top_n: int = 4
    use_reranker: bool = True

    def _build_search_retriever(self):
        dense_retriever = self.vectorstore.as_retriever(search_kwargs={'k': self.dense_k})
        bm25_retriever = _build_bm25_retriever(self.vectorstore, self.sparse_k)

        base = dense_retriever if bm25_retriever is None else EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[self.sparse_weight, self.dense_weight]
        )

        if self.use_reranker:
            reranker = _build_reranker(self.rerank_top_n)
            if reranker is not None:
                return ContextualCompressionRetriever(
                    base_compressor = reranker, base_retriever = base
                )
        return base


    def _get_relevant_documents(self,query: str, *, run_manager: CallbackManagerForRetrieverRun)->list[Document]:
        search_retriever = self._build_search_retriever()
        summary_hits = search_retriever.invoke(query)

        seen: dict[str, Document] = {}
        for doc in summary_hits:
            doc_id = doc.metadata.get(ID_KEY)
            if doc_id and doc_id not in seen:
                seen[doc_id] = doc

        if not seen:
            return []

        ordered_ids = list(seen.keys())
        originals = self.docstore.mget(ordered_ids)

        resolved: list[Document] = []
        for doc_id, original in zip(ordered_ids, originals):
            if original is None:
                continue
            resolved.append(
                Document(
                    page_content=original if isinstance(original, str) else "",
                    metadata=seen[doc_id].metadata,  # carries kind/source/page for citations
                )
            )
        return resolved



def get_retriever(
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    use_reranker: bool = True,
) -> HybridMultiVectorRetriever:
    settings = get_settings()
    return HybridMultiVectorRetriever(
        vectorstore=_get_vectorstore(),
        docstore=get_docstore(),
        dense_k=settings.retrieval_k,
        sparse_k=settings.retrieval_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        rerank_top_n=settings.retrieval_k,
        use_reranker=use_reranker,
    )

def index_elements(
    retriever: HybridMultiVectorRetriever,
    originals: list[str],
    summaries: list[str],
    kind: str,
    source_pdf: str,
    pages: list[int] | None = None,
) -> None:
    """Index one element type (text/table/image) into the retriever.
    `originals` = the raw content handed to the LLM at answer time
    `summaries` = the (usually shorter, LLM-generated) text embedded for search
    Unchanged from before -- BM25 is rebuilt from the vectorstore on read,
    so nothing extra needs to happen here at index time.
    """
    if not originals:
        return
    pages = pages or [0] * len(originals)
    doc_ids = [str(uuid.uuid4()) for _ in originals]
    summary_docs = [
        Document(
            page_content=summary,
            metadata={ID_KEY: doc_id, "kind": kind, "source": source_pdf, "page": page},
        )
        for doc_id, summary, page in zip(doc_ids, summaries, pages)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, originals)))
    logger.info("Indexed %d %s elements from %s", len(originals), kind, source_pdf)



def get_retrieved_metadata(retriever: HybridMultiVectorRetriever, query: str) -> list[Document]:
    """Return the resolved Documents (original content + kind/source/page metadata)
    for a query -- exactly what generation sees, so citations built from this are
    guaranteed consistent with the answer.
    """
    return retriever.invoke(query)




def reset_index() -> None:
    """Wipe everything previously indexed — the Chroma collection AND the
    docstore of original text/table/image content.

    This app runs in single-active-document mode: uploading a new PDF is
    meant to fully replace the previous one, not add to it. Only clearing
    the uploaded file on disk isn't enough for that — the vectorstore and
    docstore would still happily answer questions from the old PDF, since
    they're keyed by content, not by "current file". This clears both so
    a fresh upload starts from a genuinely empty index.
    """
    vectorstore = _get_vectorstore()
    try:
        vectorstore.delete_collection()
    except Exception:
        logger.exception("Failed to delete Chroma collection during index reset")

    docstore = get_docstore()
    keys = list(docstore.yield_keys())
    if keys:
        docstore.mdelete(keys)

    logger.info("Index reset: cleared Chroma collection and %d docstore entries", len(keys))
