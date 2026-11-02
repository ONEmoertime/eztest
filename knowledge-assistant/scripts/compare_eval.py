# -*- coding: utf-8 -*-
"""对比 .index 下多份评估结果：汇总 + 逐题 + 失败原因。

用法：python scripts/compare_eval.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

INDEX = Path(__file__).resolve().parent.parent / ".index"


def load(tag: str | None = None) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for path in sorted(INDEX.glob("eval-results-*.json")):
        key = path.stem.replace("eval-results-", "")
        if tag and not key.endswith("-" + tag):
            continue
        runs[key] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else None
    runs = load(tag)
    if not runs:
        print("没有找到评估结果，先跑 python -m ka.cli eval（可加数据集名过滤，如：compare_eval.py hard）")
        return

    keys = list(runs)
    print("评估结果对比（同一数据集）：\n")
    print(
        f"{'运行':<44}{'召回':>6}{'MRR':>7}{'覆盖':>7}{'通过':>6}{'拒答':>6}{'P95(ms)':>10}"
    )
    for key in keys:
        s = runs[key]["summary"]
        print(
            f"{key:<44}{fmt(s.get('recall@k')):>6}{fmt(s.get('mrr')):>7}"
            f"{fmt(s.get('coverage')):>7}{fmt(s.get('correct_rate')):>6}"
            f"{fmt(s.get('refusal_acc')):>6}{fmt(s.get('p95_latency_ms')):>10}"
        )

    print("\n逐题（ok = 确定性判定是否通过）：")
    by_question: dict[str, dict[str, dict]] = defaultdict(dict)
    for key in keys:
        for row in runs[key]["rows"]:
            by_question[row["question"]][key] = row

    for question, per_run in by_question.items():
        cells = []
        for key in keys:
            row = per_run.get(key)
            if row is None:
                cells.append("—")
                continue
            flag = "ok" if row.get("ok") else "FAIL"
            cells.append(f"{flag} r{row.get('rank') or '-'} cov{row.get('coverage')}")
        print(f"  {question[:30]:<32}" + " | ".join(cells))

    print("\n失败明细：")
    found = False
    for question, per_run in by_question.items():
        for key, row in per_run.items():
            if row.get("ok"):
                continue
            found = True
            reasons = []
            if row.get("missing"):
                reasons.append("缺：" + "/".join(row["missing"]))
            if row.get("forbidden"):
                reasons.append("禁词：" + "/".join(row["forbidden"]))
            if row.get("should_answer") and row.get("refused"):
                reasons.append("该答却拒答")
            if not row.get("should_answer") and not row.get("refused"):
                reasons.append("该拒答却作答")
            if row.get("recall_hit") is False:
                reasons.append("未召回目标文档")
            print(f"  [{key}] {question}\n      {'；'.join(reasons) or '综合不通过'}")
            print(f"      答：{row['answer'][:100]}")
    if not found:
        print("  无")


if __name__ == "__main__":
    main()
