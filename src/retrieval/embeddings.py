from __future__ import annotations

import logging

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once, reused everywhere
_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = settings.model.embedding_model

    logger.info(
        "Loading embedding model '%s' on %s …", model_name, device.upper()
    )
    _embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.retrieval.batch_size,
        },
    )
    logger.info("Embedding model ready.")
    return _embedding_model
