"""全局配置：从 .env 读取，缺省值保证 demo 能跑起来。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class Settings:
    api_base: str
    api_key: str
    chat_model: str
    embed_model: str
    embed_dim: int
    rerank_model: str
    top_k: int
    pool_size: int
    min_score: float
    qdrant_url: str
    collection: str
    vector_backend: str
    raw_dir: Path
    index_dir: Path


@lru_cache(maxsize=1)
def settings() -> Settings:
    _load_env(ROOT / ".env")
    return Settings(
        api_base=os.environ.get("API_BASE", "https://api.deepseek.com/v1"),
        api_key=os.environ.get("API_KEY", ""),
        chat_model=os.environ.get("CHAT_MODEL", "deepseek-chat"),
        embed_model=os.environ.get("EMBED_MODEL", "BAAI/bge-m3"),
        embed_dim=int(os.environ.get("EMBED_DIM", "1024")),
        rerank_model=os.environ.get("RERANK_MODEL", ""),
        top_k=int(os.environ.get("TOP_K", "5")),
        pool_size=int(os.environ.get("POOL_SIZE", "30")),
        min_score=float(os.environ.get("MIN_SCORE", "0.45")),
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        collection=os.environ.get("COLLECTION", "handbook"),
        vector_backend=os.environ.get("VECTOR_BACKEND", "auto"),
        raw_dir=ROOT / os.environ.get("RAW_DIR", "data/raw"),
        index_dir=ROOT / os.environ.get("INDEX_DIR", ".index"),
    )
