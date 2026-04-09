from __future__ import annotations

import logging
import os

import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Module-level singleton
_llm: HuggingFacePipeline | None = None


def get_llm() -> HuggingFacePipeline:
    global _llm
    if _llm is not None:
        return _llm

    cfg       = settings.model
    hf_token  = settings.huggingface_token

    if not hf_token:
        logger.warning(
            "HF_TOKEN not set.  Gated models (e.g. Llama 3) will fail. "
            "Add your token to .env or set the HF_TOKEN environment variable."
        )

    # ── 4-bit quantisation config ─────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading tokeniser for '%s' …", cfg.llm_model)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.llm_model,
        token=hf_token,
        padding_side="left",
    )
    # Llama 3.x uses the EOS token as the pad token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading model '%s' (4-bit) …", cfg.llm_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.llm_model,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model.eval()

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        repetition_penalty=cfg.repetition_penalty,
        do_sample=True,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    _llm = HuggingFacePipeline(pipeline=hf_pipeline)
    logger.info("LLM ready: %s", cfg.llm_model)
    return _llm
