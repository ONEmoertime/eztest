"""RRF 融合：只依赖排名，不依赖分数量纲。

单独成文件是为了让核心算法不依赖网络库，单元测试可以裸跑。
"""
from __future__ import annotations

from typing import Sequence


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion，把多路召回结果融合成一个排名。"""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)
