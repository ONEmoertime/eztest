"""问答主链路：检索 → 相关性闸门 → 生成。

加闸门的原因（来自评估数据）：向量检索在“资料里没有答案”的问题上会硬答——
4 条陷阱题里有 2 条编了答案。闸门在生成之前用相似度拦一道，
比在提示词里反复强调“不知道就说不知道”更可靠。
"""
from __future__ import annotations

from .answer import answer as generate
from .config import Settings
from .llm import LLMClient
from .retrieve import search_scored
from .store import BM25Store

REFUSAL = "根据现有资料无法回答：检索到的内容与问题相关度不足（最高相似度 {score:.2f}），资料中可能没有相关信息。"


def ask(
    cfg: Settings,
    llm: LLMClient,
    vector_store,
    bm25: BM25Store,
    question: str,
    top_k: int | None = None,
    mode: str = "hybrid",
) -> dict:
    chunks, best = search_scored(llm, vector_store, bm25, cfg, question, top_k, mode)

    gated = (
        mode != "bm25"
        and best is not None
        and best < cfg.min_score
    )
    if gated:
        return {
            "answer": REFUSAL.format(score=best),
            "used_sources": [],
            "sources": [],
            "raw": "",
            "gated": True,
            "best_score": best,
            "chunks": chunks,
        }

    result = generate(llm, question, chunks)
    result.update({"gated": False, "best_score": best, "chunks": chunks})
    return result
