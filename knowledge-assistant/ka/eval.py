"""评估：以确定性指标为主，LLM 裁判只作辅助。

指标分三层（面试时按这三层讲）：
1. 检索层（确定性）：recall@k、MRR —— 正确文档有没有召回、排在第几；
2. 生成层（确定性）：要点覆盖率 —— 标准答案里的关键要点，回答覆盖了几个；
3. 质量层（可选）：LLM 裁判 —— 只在 --judge 时启用，且单独统计，不污染主指标。

为什么这么改：早期把评分全交给裁判，结果同一份回答在不同模式下被判 4/5 分波动，
一条题的波动就改变整体结论。确定性指标能让同样的输入永远得到同样的分数。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from statistics import mean

from .answer import answer
from .config import Settings
from .ingest import Chunk
from .llm import LLMClient
from .pipeline import ask
from .retrieve import search
from .store import BM25Store
from .vector import open_vector_store

JUDGE_SYSTEM = """你是严格的评审。根据【问题】【标准答案】【模型回答】给出评分：
5=完全正确且引用恰当；4=正确但略有遗漏；3=部分正确；2=有明显错误；1=答非所问或编造。
只输出 JSON：{"score": 1-5, "reason": "一句话理由"}"""

PASS_COVERAGE = 0.8
_FULLWIDTH = {ord(c): str(i) for i, c in enumerate("０１２３４５６７８９")}
_STRIP = re.compile(r"[\s，。、；：（）()【】\[\]“”\"'·]+")


def _norm(text: str) -> str:
    """归一化：去空白与标点、全角数字转半角、转小写。"""
    return _STRIP.sub("", text.translate(_FULLWIDTH)).lower()


def _hit(point, answer_norm: str) -> bool:
    """要点命中判定。支持同义词组：["严重违纪", "解除劳动合同"] 表示任一命中即可。"""
    if isinstance(point, (list, tuple)):
        return any(_norm(str(alt)) in answer_norm for alt in point)
    return _norm(str(point)) in answer_norm


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def judge(llm: LLMClient, question: str, expected: str, got: str) -> dict:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"【问题】{question}\n【标准答案】{expected}\n【模型回答】{got}"},
    ]
    try:
        return llm.chat_json(messages, max_tokens=300)
    except Exception as exc:  # noqa: BLE001
        return {"score": 0, "reason": f"judge failed: {exc}"}


def run(
    cfg: Settings,
    cases_path: Path,
    out_path: Path | None = None,
    mode: str = "hybrid",
    use_judge: bool = False,
) -> dict:
    llm = LLMClient(cfg)
    qdrant = open_vector_store(cfg, cfg.embed_dim)
    bm25 = BM25Store(cfg)
    if not bm25.load():
        raise RuntimeError("BM25 索引不存在，请先执行 python -m ka.cli ingest")

    rows = []
    for case in load_cases(cases_path):
        started = time.perf_counter()
        result = ask(cfg, llm, qdrant, bm25, case["question"], mode=mode)
        chunks: list[Chunk] = result.get("chunks", [])
        latency = (time.perf_counter() - started) * 1000

        # ---- 检索层 ----
        expected_docs = set(case.get("sources") or [])
        rank = None
        if expected_docs:
            rank = next((i for i, c in enumerate(chunks, 1) if c.doc in expected_docs), None)
        recall_hit = (rank is not None) if expected_docs else None
        mrr = (1.0 / rank) if rank else (0.0 if expected_docs else None)

        # ---- 生成层 ----
        must = case.get("must_include") or []
        answer_norm = _norm(result["answer"])
        found = [p for p in must if _hit(p, answer_norm)]
        missing = [p for p in must if p not in found]
        coverage = (len(found) / len(must)) if must else None
        forbidden = [
            p
            for p in (case.get("must_not_include") or [])
            if _norm(p) in answer_norm and not _looks_refused(result["answer"])
        ]

        should_answer = case.get("should_answer", True)
        refused = _looks_refused(result["answer"])

        # ---- 综合判定（确定性）----
        ok = True
        if forbidden:
            ok = False
        if should_answer and refused:
            ok = False
        if not should_answer and not refused:
            ok = False
        if should_answer and coverage is not None and coverage < PASS_COVERAGE:
            ok = False
        if expected_docs and not recall_hit:
            ok = False

        row = {
            "id": case.get("id", ""),
            "question": case["question"],
            "answer": result["answer"],
            "recall_hit": recall_hit,
            "rank": rank,
            "mrr": mrr,
            "coverage": coverage,
            "missing": missing,
            "forbidden": forbidden,
            "should_answer": should_answer,
            "refused": refused,
            "ok": ok,
            "latency_ms": round(latency, 1),
        }
        if use_judge:
            verdict = judge(llm, case["question"], case.get("expected", ""), result["answer"])
            row["judge_score"] = verdict.get("score", 0)
            row["judge_reason"] = verdict.get("reason", "")
        rows.append(row)

    recalls = [r["recall_hit"] for r in rows if r["recall_hit"] is not None]
    mrrs = [r["mrr"] for r in rows if r["mrr"] is not None]
    coverages = [r["coverage"] for r in rows if r["coverage"] is not None]
    latencies = sorted(r["latency_ms"] for r in rows)
    judges = [r.get("judge_score", 0) for r in rows if use_judge]

    summary = {
        "cases": len(rows),
        "recall@k": round(mean(recalls), 4) if recalls else None,
        "mrr": round(mean(mrrs), 4) if mrrs else None,
        "coverage": round(mean(coverages), 4) if coverages else None,
        "correct_rate": round(mean([r["ok"] for r in rows]), 4) if rows else None,
        "refusal_acc": _refusal_accuracy(rows),
        "over_refusal": sum(1 for r in rows if r["should_answer"] and r["refused"]),
        "forbidden_violations": sum(1 for r in rows if r["forbidden"]),
        "p95_latency_ms": _p95(latencies),
        "avg_latency_ms": round(mean(latencies), 1) if latencies else None,
        "retrieval": mode,
        "judge_avg": round(mean(judges), 2) if judges else None,
        "usage": llm.usage,
    }

    payload = {"summary": summary, "rows": rows}
    out_path = out_path or (
        cfg.index_dir / f"eval-results-{_slug(cfg.chat_model)}-{mode}-{cases_path.stem}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n明细已写入 {out_path}")

    bad = [r for r in rows if not r["ok"]]
    if bad:
        print(f"\n未通过 {len(bad)} 条：")
        for r in bad[:5]:
            detail = []
            if r["missing"]:
                detail.append("缺要点:" + "/".join(r["missing"]))
            if r["forbidden"]:
                detail.append("出现禁词:" + "/".join(r["forbidden"]))
            if r["should_answer"] and r["refused"]:
                detail.append("该答却拒答")
            if not r["should_answer"] and not r["refused"]:
                detail.append("该拒答却作答")
            if r["recall_hit"] is False:
                detail.append("未召回目标文档")
            print(f"- {r['question']} → {'；'.join(detail) or '综合不通过'}")
    return payload


def _looks_refused(text: str) -> bool:
    """只看开头 60 字：拒答的标志性语句通常在句首，避免正文里出现“未提及”被误判。"""
    markers = ("无法回答", "资料中没有", "没有提到", "未提及", "无法确定", "不足以回答")
    head = text.strip()[:60]
    return any(marker in head for marker in markers)


def _refusal_accuracy(rows: list[dict]) -> float | None:
    traps = [r for r in rows if r.get("should_answer") is False]
    if not traps:
        return None
    return round(sum(1 for r in traps if r["refused"]) / len(traps), 4)


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    index = min(len(values) - 1, int(round(0.95 * (len(values) - 1))))
    return round(values[index], 1)


def _slug(model: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in model)
