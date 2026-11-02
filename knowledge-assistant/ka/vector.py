"""向量存储的两种后端：Qdrant（生产）与本地文件（零依赖，跑通第一步用）。

auto 模式先探测 Qdrant，连不上就自动退回本地，避免"为了跑个 demo 先装 Docker"。
"""
from __future__ import annotations

import json
import math

import requests

from .config import Settings
from .ingest import Chunk
from .store import QdrantStore


class LocalVectorStore:
    """把向量存成 JSON，检索时暴力算余弦。

    15~5000 条片段的规模完全够用；再大就该换 Qdrant 了。
    """

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.path = cfg.index_dir / "vectors.json"
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        self._loaded = False

    def ensure(self, dim: int) -> None:
        self.cfg.index_dir.mkdir(parents=True, exist_ok=True)

    def recreate(self, dim: int) -> None:
        self.ensure(dim)
        if self.path.exists():
            self.path.unlink()
        self.chunks, self.vectors, self._loaded = [], [], True

    def upsert(self, chunks, vectors) -> None:
        self._load()
        for chunk, vector in zip(chunks, vectors):
            self.chunks.append(chunk)
            self.vectors.append(list(vector))
        self._save()

    def search(self, vector: list[float], limit: int) -> list[tuple[Chunk, float]]:
        self._load()
        scored = []
        for chunk, vec in zip(self.chunks, self.vectors):
            score = _cosine(vector, vec)
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    # ---------- 持久化 ----------
    def _load(self) -> None:
        if self._loaded or not self.path.exists():
            self._loaded = True
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**c) for c in data["chunks"]]
        self.vectors = data["vectors"]
        self._loaded = True

    def _save(self) -> None:
        self.ensure(len(self.vectors[0]) if self.vectors else 0)
        payload = {
            "chunks": [c.to_dict() for c in self.chunks],
            "vectors": self.vectors,
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def open_vector_store(cfg: Settings, dim: int):
    backend = (cfg.vector_backend or "auto").lower()

    if backend == "local":
        store = LocalVectorStore(cfg)
        store.ensure(dim)
        return store

    if backend == "qdrant":
        store = QdrantStore(cfg)
        store.ensure(dim)
        return store

    # auto：探测 Qdrant
    try:
        requests.get(cfg.qdrant_url, timeout=2)
        store = QdrantStore(cfg)
        store.ensure(dim)
        print("[vector] 使用 Qdrant")
        return store
    except Exception:  # noqa: BLE001
        store = LocalVectorStore(cfg)
        store.ensure(dim)
        print("[vector] Qdrant 未启动，改用本地文件索引（.index/vectors.json）")
        return store
