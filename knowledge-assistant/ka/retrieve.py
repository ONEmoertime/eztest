"""检索层：向量 + 关键词的混合召回，RRF 融合，可选重排。

为什么要混合：向量擅长语义（“报销多久到账”≈“费用结算周期”），
BM25 擅长精确词（“IT-2024-017”这种编号、专有名词）。
只用其中一种，都会在评估集上留下明显的 badcase。
"""
from __future__ import annotations

from typing import Sequence

import requests

from .config import Settings
from .fusion import rrf
from .ingest import Chunk
from .llm import LLMClient
from .store import BM25Store, QdrantStore


def search_scored(
    llm: LLMClient,
    qdrant: QdrantStore,
    bm25: BM25Store,
    cfg: Settings,
    query: str,
    top_k: int | None = None,
    mode: str = "hybrid",
) -> tuple[list[Chunk], float | None]:
    """返回（片段列表, 最高向量相似度）。相似度用于拒答闸门：太低就不该让模型硬答。"""
    top_k = top_k or cfg.top_k
    pool = cfg.pool_size
    mode = (mode or "hybrid").lower()

    vec_hits: list[tuple[Chunk, float]] = []
    kw_hits: list[tuple[Chunk, float]] = []

    if mode in ("hybrid", "vector"):
        vector = llm.embed([query])[0]
        vec_hits = qdrant.search(vector, pool)
    if mode in ("hybrid", "bm25"):
        kw_hits = bm25.search(query, pool)

    best_score = max((score for _, score in vec_hits), default=None)

    by_id: dict[str, Chunk] = {c.id: c for c, _ in vec_hits}
    by_id.update({c.id: c for c, _ in kw_hits})

    if mode == "vector":
        candidates = [c for c, _ in vec_hits][:pool]
    elif mode == "bm25":
        candidates = [c for c, _ in kw_hits][:pool]
    else:
        fused_ids = rrf([[c.id for c, _ in vec_hits], [c.id for c, _ in kw_hits]])
        candidates = [by_id[i] for i in fused_ids if i in by_id][:pool]

    candidates = rerank(cfg, query, candidates)
    return candidates[:top_k], best_score


def search(
    llm: LLMClient,
    qdrant: QdrantStore,
    bm25: BM25Store,
    cfg: Settings,
    query: str,
    top_k: int | None = None,
    mode: str = "hybrid",
) -> list[Chunk]:
    """mode: hybrid（向量+BM25+RRF）/ vector（纯向量）/ bm25（纯关键词）。"""
    chunks, _ = search_scored(llm, qdrant, bm25, cfg, query, top_k, mode)
    return chunks


def rerank(cfg: Settings, query: str, chunks: list[Chunk]) -> list[Chunk]:
    """可选重排：配置了 RERANK_MODEL 且服务商支持 /rerank 时启用，否则原样返回。"""
    if not cfg.rerank_model or len(chunks) <= 1:
        return chunks
    url = cfg.api_base.rstrip("/") + "/rerank"
    payload = {
        "model": cfg.rerank_model,
        "query": query,
        "documents": [c.text for c in chunks],
        "top_n": len(chunks),
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json().get("results") or data_fallback(resp)
        ordered = []
        for item in sorted(data, key=lambda d: d.get("relevance_score", 0), reverse=True):
            index = item.get("index")
            if index is not None and 0 <= index < len(chunks):
                ordered.append(chunks[index])
        return ordered or chunks
    except Exception:  # noqa: BLE001  重排失败不能影响主链路
        return chunks


def data_fallback(resp) -> list[dict]:
    try:
        return resp.json().get("data", [])
    except Exception:  # noqa: BLE001
        return []
