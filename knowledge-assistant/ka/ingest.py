"""文档解析与切片。

设计取舍（面试会问）：
1. 切片优先按标题层级，其次按段落，最后按句子——保证引用时能定位到章节；
2. 每个 chunk 带 doc/page/section 元数据，引用才能落到具体位置；
3. 中文 BM25 没有分词器时用「汉字二元组」，够用且零依赖。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".md", ".txt"}


@dataclass
class Page:
    text: str
    doc: str
    page: int


@dataclass
class Chunk:
    id: str
    text: str
    doc: str
    page: int
    section: str

    def to_dict(self) -> dict:
        return asdict(self)


def parse_file(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        return []
    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix == ".docx":
        return _parse_docx(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Page(text=text, doc=path.name, page=1)]


def _parse_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Page(text=text, doc=path.name, page=i))
    return pages


def _parse_docx(path: Path) -> list[Page]:
    from docx import Document

    doc = Document(str(path))
    text = "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    return [Page(text=text, doc=path.name, page=1)]


def iter_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []
    return sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in SUPPORTED and p.is_file())


def split_text(text: str, size: int = 700, overlap: int = 120) -> list[str]:
    """按段落贪心合并，超长段落再按句子切。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > size:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_split_long(para, size, overlap))
            continue
        if len(buf) + len(para) + 2 <= size:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    # 加重叠：把上一块尾部拼到下一块开头，缓解跨块语义割裂
    if overlap > 0 and len(chunks) > 1:
        merged = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            tail = prev[-overlap:]
            merged.append(tail + "\n" + cur)
        chunks = merged
    return chunks


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    sentences = re.split(r"(?<=[。！？.!?])\s*", text)
    out, buf = [], ""
    for sent in sentences:
        if not sent:
            continue
        if len(buf) + len(sent) <= size:
            buf += sent
        else:
            if buf:
                out.append(buf)
            buf = sent[-size:] if len(sent) > size else sent
    if buf:
        out.append(buf)
    return out


def chunk_pages(pages: list[Page], size: int = 700, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    counter = 0
    for page in pages:
        section = ""
        current: list[str] = []

        def flush() -> None:
            nonlocal counter, current
            if not current:
                return
            body = "\n\n".join(current)
            for piece in split_text(body, size, overlap):
                counter += 1
                chunks.append(
                    Chunk(
                        id=f"{page.doc}#p{page.page}#c{counter}",
                        text=piece.strip(),
                        doc=page.doc,
                        page=page.page,
                        section=section,
                    )
                )
            current = []

        for line in page.text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                flush()
                section = stripped.lstrip("#").strip()
                continue
            if stripped:
                current.append(stripped)
        flush()
    return chunks


def tokenize(text: str) -> list[str]:
    """英文按词、中文按二元组，零依赖的 BM25 分词方案。"""
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.extend("".join(pair) for pair in zip(chinese, chinese[1:]))
    return tokens
