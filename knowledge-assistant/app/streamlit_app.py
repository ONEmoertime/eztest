"""Streamlit 演示界面：问答 + 引用展开 + 检索耗时。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 让 streamlit 从仓库根目录导入 ka 包
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from ka.pipeline import ask
from ka.config import settings
from ka.llm import LLMClient
from ka.store import BM25Store
from ka.vector import open_vector_store

st.set_page_config(page_title="企业知识库助手", page_icon="📚", layout="wide")


@st.cache_resource
def resources():
    cfg = settings()
    bm25 = BM25Store(cfg)
    if not bm25.load():
        st.error("索引不存在，请先运行：python -m ka.cli ingest")
        st.stop()
    return cfg, LLMClient(cfg), open_vector_store(cfg, cfg.embed_dim), bm25


cfg, llm, qdrant, bm25 = resources()

st.title("📚 企业知识库助手")
st.caption("混合检索（向量 + BM25 + RRF）· 可引用 · 不编造")

with st.sidebar:
    st.subheader("检索参数")
    top_k = st.slider("引用片段数量 Top-K", 1, 10, cfg.top_k)
    show_sources = st.checkbox("显示全部来源", value=True)
    st.divider()
    st.caption(f"模型：{cfg.chat_model}")
    st.caption(f"向量：{cfg.embed_model}")
    st.caption(f"重排：{cfg.rerank_model or '未启用'}")

question = st.chat_input("问点什么，比如：出差住宿标准是多少？")

if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        started = time.perf_counter()
        with st.spinner("检索并生成中…"):
            result = ask(cfg, llm, qdrant, bm25, question, top_k)
        elapsed = (time.perf_counter() - started) * 1000

        st.write(result["answer"])
        gate = "（相关性闸门触发）" if result.get("gated") else ""
        st.caption(f"耗时 {elapsed:.0f} ms · 引用 {result['used_sources'] or '无'}{gate}")

        for item in result["sources"]:
            if show_sources or item["index"] in result["used_sources"]:
                label = f"[{item['index']}] {item['doc']} 第{item['page']}页"
                if item["section"]:
                    label += f" · {item['section']}"
                with st.expander(label):
                    st.write(item["text"])
