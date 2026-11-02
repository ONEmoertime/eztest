"""模型调用封装：统一重试、限流退避、token 统计、错误信息。

这是两个作品（A/B）共用的最底层，面试时值得单独讲：
把重试、超时、成本统计收在一处，上层业务代码不需要关心模型服务商。
"""
from __future__ import annotations

import json
import random
import time
from typing import Any, Sequence

import requests

from .config import Settings


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
            }
        )
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    # ---------- 底层 ----------
    def _post(self, path: str, payload: dict, retries: int = 4, timeout: int = 120) -> dict:
        if not self.cfg.api_key or self.cfg.api_key.startswith("sk-xxxx"):
            raise LLMError("未配置 API_KEY，请复制 .env.example 为 .env 并填写密钥")
        url = self.cfg.api_base.rstrip("/") + path
        last: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self.session.post(url, json=payload, timeout=timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt == retries - 1:
                    break
                # 指数退避 + 抖动：没有退避的重试等于自我 DDoS
                time.sleep(min(2 ** attempt, 8) + random.random())
        raise LLMError(f"模型调用失败：{last}")

    # ---------- 对话 ----------
    def chat(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        payload = {
            "model": self.cfg.chat_model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self._post("/chat/completions", payload)
        usage = data.get("usage") or {}
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"]

    # ---------- 向量 ----------
    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            data = self._post("/embeddings", {"model": self.cfg.embed_model, "input": batch})
            # 保证顺序与输入一致（部分服务商不保证，按 index 排序）
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
            vectors.extend(item["embedding"] for item in items)
        return vectors

    # ---------- 结构化输出 ----------
    def chat_json(self, messages: Sequence[dict[str, str]], max_tokens: int = 800) -> dict[str, Any]:
        text = self.chat(messages, temperature=0.0, max_tokens=max_tokens)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            text = text[4:] if text.startswith("json") else text
        return json.loads(text)
