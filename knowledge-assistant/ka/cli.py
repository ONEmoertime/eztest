"""命令行入口：ingest / ask / eval。"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace

from .answer import answer
from .config import settings
from .eval import run as run_eval
from .ingest import chunk_pages, iter_files, parse_file
from .llm import LLMClient
from .pipeline import ask as ask_pipeline
from .retrieve import search
from .store import BM25Store
from .vector import open_vector_store


def cmd_ingest(args: argparse.Namespace) -> None:
    cfg = settings()
    files = iter_files(cfg.raw_dir)
    if not files:
        print(f"没有找到文档，请把 PDF/DOCX/MD 放进 {cfg.raw_dir}")
        return
    print(f"发现 {len(files)} 个文档：{[f.name for f in files]}")

    pages = []
    for path in files:
        parsed = parse_file(path)
        print(f"  {path.name}: {len(parsed)} 页/段")
        pages.extend(parsed)

    chunks = chunk_pages(pages, size=args.chunk_size, overlap=args.overlap)
    print(f"切片完成：{len(chunks)} 块，平均长度 {sum(len(c.text) for c in chunks) // max(len(chunks), 1)} 字")

    llm = LLMClient(cfg)
    vectors = open_vector_store(cfg, cfg.embed_dim)
    if args.recreate:
        vectors.recreate(cfg.embed_dim)

    batch = 32
    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        batch_vectors = llm.embed([c.text for c in part])
        vectors.upsert(part, batch_vectors)
        print(f"  已写入 {min(start + batch, len(chunks))}/{len(chunks)}")

    BM25Store(cfg).build(chunks)
    print("完成：向量库与 BM25 索引均已更新")


def cmd_ask(args: argparse.Namespace) -> None:
    cfg = _with_model(settings(), getattr(args, "model", None))
    llm = LLMClient(cfg)
    qdrant = open_vector_store(cfg, cfg.embed_dim)
    bm25 = BM25Store(cfg)
    if not bm25.load():
        raise SystemExit("BM25 索引不存在，请先执行：python -m ka.cli ingest")

    started = time.perf_counter()
    result = ask_pipeline(cfg, llm, qdrant, bm25, args.question, args.top_k, args.retrieval)
    cost = (time.perf_counter() - started) * 1000

    print("\n回答：\n" + result["answer"])
    gate = "（相关性闸门触发）" if result.get("gated") else ""
    print(f"\n耗时 {cost:.0f} ms | 引用 {result['used_sources']}{gate} | 最高相似度 {result.get('best_score')}")
    print("\n来源：")
    for item in result["sources"]:
        if item["index"] in result["used_sources"] or args.all_sources:
            print(f"  [{item['index']}] {item['doc']} 第{item['page']}页 · {item['section']}")
            print(f"      {item['text'][:80]}...")


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = _with_model(settings(), getattr(args, "model", None))
    run_eval(cfg, args.file, mode=args.retrieval, use_judge=args.judge)


def _with_model(cfg, model: str | None):
    """允许单次运行覆盖对话模型，方便做 A/B 对比。"""
    return replace(cfg, chat_model=model) if model else cfg


def cmd_selftest(args: argparse.Namespace) -> None:
    """离线自检：不调用任何模型，验证解析、切片、BM25 检索是否正常。"""
    cfg = settings()
    files = iter_files(cfg.raw_dir)
    if not files:
        raise SystemExit(f"{cfg.raw_dir} 下没有文档")
    pages = []
    for path in files:
        pages.extend(parse_file(path))
    chunks = chunk_pages(pages)
    bm25 = BM25Store(cfg)
    bm25.build(chunks)

    print(f"文档 {len(files)} 个 / 切片 {len(chunks)} 块 / BM25 已建 (.index/bm25.pkl)")
    for query in ("住宿标准", "年假", "离职账号"):
        hits = bm25.search(query, 3)
        print(f"\n查询「{query}」命中 {len(hits)} 条：")
        for chunk, score in hits:
            print(f"  [{score:.2f}] {chunk.doc} · {chunk.section} · {chunk.text[:50]}")
    print("\n离线自检通过。下一步：填好 .env 后运行 python -m ka.cli ingest")


def main() -> None:
    parser = argparse.ArgumentParser(prog="ka", description="企业知识库助手")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="解析文档、建索引")
    p_ingest.add_argument("--recreate", action="store_true", help="重建 collection")
    p_ingest.add_argument("--chunk-size", type=int, default=700, help="切片最大字数，默认 700")
    p_ingest.add_argument("--overlap", type=int, default=120, help="切片重叠字数，默认 120")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="提问")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.add_argument("--model", default=None, help="覆盖对话模型，如 deepseek-ai/DeepSeek-V4.1-Flash")
    p_ask.add_argument(
        "--retrieval",
        default="hybrid",
        choices=["hybrid", "vector", "bm25"],
        help="检索模式",
    )
    p_ask.add_argument("--all-sources", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="跑评估集")
    p_eval.add_argument("--file", default="data/eval/questions.jsonl")
    p_eval.add_argument("--model", default=None, help="覆盖对话模型，结果文件按模型名区分")
    p_eval.add_argument(
        "--retrieval",
        default="hybrid",
        choices=["hybrid", "vector", "bm25"],
        help="检索模式：hybrid=向量+BM25融合，vector=纯向量，bm25=纯关键词",
    )
    p_eval.add_argument(
        "--judge",
        action="store_true",
        help="额外调用 LLM 裁判打分（默认关闭，结果单独统计，不影响主指标）",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_self = sub.add_parser("selftest", help="离线自检（不需要 API Key）")
    p_self.set_defaults(func=cmd_selftest)

    args = parser.parse_args()
    from pathlib import Path

    if getattr(args, "file", None):
        args.file = Path(args.file)
    args.func(args)


if __name__ == "__main__":
    main()
