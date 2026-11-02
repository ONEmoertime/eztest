"""生成层：把检索到的片段变成带引用的回答。

三条硬规则：
1. 只依据资料回答，资料里没有就说不知道——抑制幻觉的第一道闸门；
2. 每句话标注 [n]，n 对应 sources 序号，引用可回溯；
3. 输出结构化 JSON，便于前端渲染与评估解析。
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Sequence

from .ingest import Chunk

if TYPE_CHECKING:  # 只用于类型标注，避免测试环境被迫安装网络依赖
    from .llm import LLMClient

SYSTEM = """你是企业知识库助手，必须严格遵守以下规则：
1. 只使用【资料】中的内容回答，不得引入资料以外的知识；
2. 每个结论后面标注来源编号，格式如 [1]、[2]；
3. 如果资料不足以回答，直接回答「根据现有资料无法回答」，并说明缺少什么信息；
4. 【完整性】资料中出现的所有数字（金额、天数、期限、比例）必须原样写进回答，一个都不能少；资料给出多个条件时逐条列出，不得只答一半；
5. 回答简洁，先给结论，再给依据；
6. 只输出 JSON：{"answer": "...", "used_sources": [1,2]}"""


def build_contexts(chunks: Sequence[Chunk]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        where = f"{chunk.doc}" + (f" 第{chunk.page}页" if chunk.page > 1 else "")
        if chunk.section:
            where += f" · {chunk.section}"
        lines.append(f"[{i}] 来源：{where}\n{chunk.text}")
    return "\n\n".join(lines)


def answer(llm: "LLMClient", question: str, chunks: Sequence[Chunk]) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"【资料】\n{build_contexts(chunks)}\n\n【问题】\n{question}",
        },
    ]
    raw = llm.chat(messages, temperature=0.1, max_tokens=800)
    text, used = _parse(raw)
    sources = [
        {
            "index": i + 1,
            "doc": c.doc,
            "page": c.page,
            "section": c.section,
            "text": c.text,
        }
        for i, c in enumerate(chunks)
    ]
    return {"answer": text, "used_sources": used, "sources": sources, "raw": raw}


def _parse(raw: str) -> tuple[str, list[int]]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned[4:] if cleaned.startswith("json") else cleaned
    try:
        data = json.loads(cleaned)
        return str(data.get("answer", "")).strip(), [int(x) for x in data.get("used_sources", [])]
    except Exception:  # noqa: BLE001  模型没按格式返回时，退化为纯文本 + 正则提取引用
        used = [int(n) for n in re.findall(r"\[(\d+)\]", raw)]
        return raw.strip(), sorted(set(used))
