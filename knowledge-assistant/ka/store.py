"""存储层：Qdrant（向量） + 本地 BM25（关键词）。

用 REST 直连 Qdrant，不引入 qdrant-client，依赖更轻，也更能说明你懂接口。
"""
from __future__ import annotations

import pickle
import uuid
from pathlib import Path
from typing import Iterable

import requests

from .config import Settings
from .ingest import Chunk, tokenize

NS = uuid.UUID("6f1c3f4e-0d7a-4c1e-9a3b-1f2d3c4b5a60")


class QdrantStore:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.base = cfg.qdrant_url.rstrip("/")

    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    def ensure(self, dim: int) -> None:
        name = self.cfg.collection
        resp = requests.get(self._url(f"/collections/{name}"), timeout=10)
        if resp.status_code == 200:
            return
        body = {"vectors": {"size": dim, "distance": "Cosine"}}
        created = requests.put(self._url(f"/collections/{name}"), json=body, timeout=30)
        created.raise_for_status()

    def recreate(self, dim: int) -> None:
        requests.delete(self._url(f"/collections/{self.cfg.collection}"), timeout=30)
        self.ensure(dim)

    def upsert(self, chunks: Iterable[Chunk], vectors: list[list[float]]) -> None:
        points = []
        for chunk, vector in zip(chunks, vectors):
            points.append(
                {
                    "id": str(uuid.uuid5(NS, chunk.id)),
                    "vector": vector,
                    "payload": chunk.to_dict(),
                }
            )
        url = self._url(f"/collections/{self.cfg.collection}/points?wait=true")
        resp = requests.put(url, json={"points": points}, timeout=120)
        resp.raise_for_status()

    def search(self, vector: list[float], limit: int) -> list[tuple[Chunk, float]]:
        url = self._url(f"/collections/{self.cfg.collection}/points/search")
        payload = {"vector": vector, "limit": limit, "with_payload": True}
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        results = []
        for hit in resp.json().get("result", []):
            data = hit["payload"]
            results.append(
                (
                    Chunk(
                        id=data["id"],
                        text=data["text"],
                        doc=data["doc"],
                        page=data["page"],
                        section=data.get("section", ""),
                    ),
                    float(hit["score"]),
                )
            )
        return results


class BM25Store:
    """本地关键词索引，和向量检索互补：专有名词、编号、精确措辞靠它。"""

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.path: Path = cfg.index_dir / "bm25.pkl"
        self.chunks: list[Chunk] = []
        self._bm25 = None

    def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        corpus = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self.cfg.index_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as fh:
            pickle.dump({"chunks": [c.to_dict() for c in self.chunks], "bm25": self._bm25}, fh)

    def load(self) -> bool:
        if not self.path.exists():
            return False
        with self.path.open("rb") as fh:
            data = pickle.load(fh)
        self.chunks = [Chunk(**c) for c in data["chunks"]]
        self._bm25 = data["bm25"]
        return True

    def search(self, query: str, limit: int) -> list[tuple[Chunk, float]]:
        if not self._bm25 or not self.chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)[:limit]
        return [(chunk, float(score)) for chunk, score in ranked if score > 0]
