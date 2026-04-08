from __future__ import annotations

import gc
import logging

import faiss
import numpy as np
import torch
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

from src.config.settings import settings
from src.retrieval.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


def build_and_save_index(chunks: list[Document]) -> FAISS:
    cfg        = settings.retrieval
    embed_fn   = get_embedding_model()
    batch_size = cfg.batch_size

    texts  = [doc.page_content for doc in chunks]
    n      = len(texts)
    total_batches = (n + batch_size - 1) // batch_size

    logger.info("Computing embeddings for %d chunks in %d batches …", n, total_batches)

    raw_embeddings: list[list[float]] = []
    with tqdm(total=n, desc="Embedding", unit="doc") as pbar:
        for i in range(0, n, batch_size):
            batch      = texts[i : i + batch_size]
            batch_embs = embed_fn.embed_documents(batch)
            raw_embeddings.extend(batch_embs)
            pbar.update(len(batch))

    embeddings = np.array(raw_embeddings, dtype=np.float32)
    dimension  = embeddings.shape[1]
    logger.info("Embeddings computed — shape: %s", embeddings.shape)

    # Build raw FAISS index (inner-product; works with L2-normalised vecs)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    logger.info("FAISS index built — %d vectors", index.ntotal)

    # Wrap into LangChain FAISS so it handles docstore + retriever API
    docstore            = InMemoryDocstore({str(i): doc for i, doc in enumerate(chunks)})
    index_to_docstore   = {i: str(i) for i in range(len(chunks))}

    vectorstore = FAISS(
        embedding_function=embed_fn,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore,
    )

    # Persist to disk
    save_path = cfg.faiss_index_path
    vectorstore.save_local(save_path)
    logger.info("FAISS index saved to: %s", save_path)

    # Free temporary memory
    del raw_embeddings, embeddings
    gc.collect()
    torch.cuda.empty_cache()

    return vectorstore


def load_faiss_retriever(index_path: str | None = None):
    
    cfg  = settings.retrieval
    path = index_path or cfg.faiss_index_path
    embed_fn = get_embedding_model()

    logger.info("Loading FAISS index from: %s", path)
    vectorstore = FAISS.load_local(
        path,
        embed_fn,
        allow_dangerous_deserialization=True,  # required for pickle loading
    )
    logger.info("FAISS index loaded — %d vectors", vectorstore.index.ntotal)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":            cfg.top_k_retrieval,
            "fetch_k":      cfg.fetch_k,
            "lambda_mult":  cfg.lambda_mult,
        },
    )
    return retriever
