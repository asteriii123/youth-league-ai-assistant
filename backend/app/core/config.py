"""
项目配置中心。

负责管理：

1. 数据库配置
2. DeepSeek配置
3. Embedding模型配置
4. Rerank模型配置
5. Whisper配置
6. RAG参数配置


例如：

Embedding模型：
BAAI/bge-m3

Rerank：
bge-reranker


所有模块通过：

settings.xxx

读取配置。
"""
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", BACKEND_DIR / "data"))
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{BACKEND_DIR / 'data' / 'app.db'}")
    jwt_secret: str = os.getenv("JWT_SECRET", "local-development-change-me")
    jwt_expire_hours: int = int(os.getenv("JWT_EXPIRE_HOURS", "12"))
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "").strip()
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "").strip()
    web_search_timeout: float = float(os.getenv("WEB_SEARCH_TIMEOUT", "10"))
    web_search_provider_results: int = int(os.getenv("WEB_SEARCH_PROVIDER_RESULTS", "5"))
    web_search_max_results: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "8"))
    modelscope_api_token: str = os.getenv("MODELSCOPE_API_TOKEN", "").strip()
    modelscope_base_url: str = os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1").rstrip("/")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    embedding_model: str = os.getenv("EMBEDDING_MODEL", os.getenv("MODELSCOPE_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"))
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", os.getenv("MODELSCOPE_EMBEDDING_BATCH_SIZE", "16")))
    embedding_retries: int = int(os.getenv("EMBEDDING_RETRIES", os.getenv("MODELSCOPE_EMBEDDING_RETRIES", "3")))
    rerank_provider: str = os.getenv("RERANK_PROVIDER", "local").strip().lower()
    rerank_model: str = os.getenv("RERANK_MODEL", os.getenv("MODELSCOPE_RERANK_MODEL", "BAAI/bge-reranker-base"))
    rerank_device: str = os.getenv("RERANK_DEVICE", "cpu")
    rerank_batch_size: int = int(os.getenv("RERANK_BATCH_SIZE", "4"))
    rag_enabled: bool = os.getenv("RAG_ENABLED", "true").lower() == "true"
    whisper_enabled: bool = os.getenv("WHISPER_ENABLED", "true").lower() == "true"
    whisper_model: str = os.getenv("WHISPER_MODEL", "small")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    rag_final_top_k: int = int(os.getenv("RAG_FINAL_TOP_K", "3"))
    rag_recall_top_k: int = int(os.getenv("RAG_RECALL_TOP_K", "50"))
    rag_rrf_k: int = int(os.getenv("RAG_RRF_K", "60"))
    rag_fusion_top_k: int = int(os.getenv("RAG_FUSION_TOP_K", "20"))
    rag_min_rerank_score: float = float(os.getenv("RAG_MIN_RERANK_SCORE", "0.35"))

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def converted_dir(self) -> Path:
        return self.data_dir / "converted"

    @property
    def chroma_dir(self) -> Path:
        return Path(os.getenv("CHROMA_PERSIST_DIR", self.data_dir / "chroma"))

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def models_dir(self) -> Path:
        return Path(os.getenv("LOCAL_MODEL_CACHE_DIR", self.data_dir / "models"))

    @property
    def tessdata_dir(self) -> Path:
        return Path(os.getenv("TESSDATA_DIR", self.data_dir / "tessdata"))

    @property
    def langgraph_checkpoint_path(self) -> Path:
        return self.data_dir / "langgraph" / "checkpoints.sqlite"

    @property
    def meeting_documents_dir(self) -> Path:
        return self.data_dir / "meeting_documents"


settings = Settings()


def ensure_data_directories() -> None:
    for path in (
        settings.data_dir,
        settings.converted_dir,
        settings.chroma_dir,
        settings.indexes_dir,
        settings.models_dir,
        settings.uploads_dir / "notices",
        settings.uploads_dir / "submissions",
        settings.uploads_dir / "meetings",
        settings.uploads_dir / "knowledge",
        settings.tessdata_dir,
        settings.langgraph_checkpoint_path.parent,
        settings.meeting_documents_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def secure_filename_suffix() -> str:
    return secrets.token_hex(8)
