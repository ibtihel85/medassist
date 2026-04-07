import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    llm_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    load_in_4bit: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.1
    repetition_penalty: float = 1.15


@dataclass
class RetrievalConfig:
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
    top_k_retrieval: int = 5
    fetch_k: int = 20
    lambda_mult: float = 0.7
    chunk_size: int = 512
    chunk_overlap: int = 64
    batch_size: int = 256


@dataclass
class ConfidenceConfig:
    threshold: float = 0.40
    refuse: float = 0.20

@dataclass
class AgentConfig:
    max_retries: int = 2

@dataclass
class Settings:
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    huggingface_token: str = os.getenv("HF_TOKEN", "")

    device: str = "auto"


settings = Settings()
