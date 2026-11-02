"""核心逻辑的单元测试（不依赖模型与数据库）。

运行：python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ka.answer import build_contexts, _parse
from ka.fusion import rrf
from ka.ingest import Chunk, Page, chunk_pages, split_text, tokenize


class TestSplit(unittest.TestCase):
    def test_short_text_single_chunk(self):
        self.assertEqual(len(split_text("短文本", 700, 120)), 1)

    def test_long_text_respects_size(self):
        text = "。".join(["这是一句用于测试的长文本"] * 200)
        chunks = split_text(text, 200, 20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) < 400 for c in chunks))


class TestChunking(unittest.TestCase):
    def test_section_is_tracked(self):
        page = Page(text="# 报销制度\n\n差旅费按标准报销。\n\n## 住宿\n\n一线城市 600 元。", doc="a.md", page=1)
        chunks = chunk_pages([page], size=200, overlap=0)
        self.assertTrue(any(c.section == "报销制度" for c in chunks))
        self.assertTrue(all(c.doc == "a.md" for c in chunks))
        self.assertEqual(len({c.id for c in chunks}), len(chunks))


class TestRRF(unittest.TestCase):
    def test_fusion_prefers_docs_in_both_lists(self):
        fused = rrf([["a", "b", "c"], ["c", "a", "d"]])
        self.assertEqual(fused[0], "a")

    def test_single_list_preserves_order(self):
        self.assertEqual(rrf([["x", "y"]]), ["x", "y"])


class TestTokenize(unittest.TestCase):
    def test_chinese_bigrams(self):
        tokens = tokenize("报销标准")
        self.assertIn("报销", tokens)
        self.assertIn("标准", tokens)

    def test_english_words(self):
        self.assertIn("api", tokenize("API latency"))


class TestAnswerParsing(unittest.TestCase):
    def test_json_output(self):
        text, used = _parse('{"answer": "600 元 [1]", "used_sources": [1]}')
        self.assertEqual(text, "600 元 [1]")
        self.assertEqual(used, [1])

    def test_fallback_plain_text(self):
        text, used = _parse("一线城市 600 元 [2]，二线 400 元 [1]")
        self.assertEqual(used, [1, 2])

    def test_contexts_include_source_metadata(self):
        chunk = Chunk(id="1", text="正文", doc="制度.md", page=3, section="住宿")
        ctx = build_contexts([chunk])
        self.assertIn("制度.md", ctx)
        self.assertIn("第3页", ctx)
        self.assertIn("[1]", ctx)


if __name__ == "__main__":
    unittest.main()
