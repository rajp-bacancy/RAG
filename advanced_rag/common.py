"""Shared building blocks: PDF text, hybrid (dense + BM25) index, Gemini generation."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "indexes"

EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


# ---------------------------------------------------------------- PDF text

def pdf_pages(path: Path, layout: bool = False) -> List[str]:
    """Per-page text via poppler's pdftotext (index 0 = PDF page 1)."""
    cmd = ["pdftotext"] + (["-layout"] if layout else []) + [str(path), "-"]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout.decode("utf-8", "replace")
    pages = out.split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return pages


def normalize(text: str) -> str:
    """NFKC folds ligatures (ﬁ -> fi); drop soft hyphens and odd whitespace."""
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    return text.replace(" ", " ")


def words(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------- BM25

_STOP = set(
    "a an and are as at be by for from has have how in is it its of on or that the "
    "this to was were what which who whom with does do did much many about".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenize(text: str) -> List[str]:
    toks = [t.replace(",", "") for t in _TOKEN.findall(text.lower())]
    return [t for t in toks if t not in _STOP]


class BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avg = self.len.mean() if len(docs) else 1.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.tf))
        for i, c in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            s = 0.0
            for t in q:
                f = c.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            out[i] = s
        return out


# ---------------------------------------------------------------- embeddings

_embedder = None


def embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def embed(texts: List[str], batch_size: int = 64) -> np.ndarray:
    return embedder().encode(
        texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=len(texts) > 200
    )


# ---------------------------------------------------------------- hybrid index

class Index:
    """Chunks + dense embeddings + BM25. Each chunk is a dict with at least
    `text` (shown to the LLM), `embed_text` (what is embedded/BM25'd: text prefixed
    with its section breadcrumb) and `page_start`/`page_end` (printed page labels)."""

    def __init__(self, chunks: List[dict], embeddings: np.ndarray):
        self.chunks = chunks
        self.emb = embeddings
        self._bm25: Optional[BM25] = None

    @classmethod
    def build(cls, chunks: List[dict]) -> "Index":
        for i, c in enumerate(chunks):
            c["id"] = i
        return cls(chunks, embed([c["embed_text"] for c in chunks]))

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([c["embed_text"] for c in self.chunks])
        return self._bm25

    def save(self, name: str) -> None:
        INDEX_DIR.mkdir(exist_ok=True)
        np.save(INDEX_DIR / f"{name}.npy", self.emb)
        (INDEX_DIR / f"{name}.json").write_text(json.dumps(self.chunks, ensure_ascii=False))

    @classmethod
    def load(cls, name: str) -> "Index":
        p = INDEX_DIR / f"{name}.json"
        if not p.exists():
            raise FileNotFoundError(f"No index '{name}'. Run: python ingest_advanced.py {name}")
        return cls(json.loads(p.read_text()), np.load(INDEX_DIR / f"{name}.npy"))

    def search(
        self,
        query: str,
        k: int = 6,
        allow: Optional[Callable[[dict], bool]] = None,
        boost: Optional[Callable[[dict], float]] = None,
        pool: int = 40,
    ) -> List[dict]:
        """Hybrid retrieval: reciprocal-rank fusion of dense + BM25 over the allowed chunks."""
        ids = [c["id"] for c in self.chunks if allow is None or allow(c)]
        if not ids:
            return []
        idx = np.array(ids)
        qv = embed([query])[0]
        dense = self.emb[idx] @ qv
        sparse = self.bm25.scores(query)[idx]
        fused = np.zeros(len(idx))
        for s in (dense, sparse):
            if s.max() <= 0:
                continue
            for rank, j in enumerate(np.argsort(-s)[:pool]):
                fused[j] += 1.0 / (60 + rank)
        if boost:
            fused = fused + np.array([boost(self.chunks[i]) for i in ids]) / 60.0
        order = np.argsort(-fused)[:k]
        return [dict(self.chunks[ids[j]], score=float(fused[j]), dense=float(dense[j])) for j in order]


# ---------------------------------------------------------------- generation

BASE_RULES = (
    "Answer ONLY from the numbered context excerpts below. If the excerpts do not contain the "
    "answer, say you could not find it in the document - never use outside knowledge or guess. "
    "Quote exact figures as written (with units and years). Cite the page(s) you used like (p. 12) using the page numbers in the excerpt headers, never the excerpt numbers. "
    "Be concise."
)


def call_gemini(system: str, user: str, retries: int = 4, max_tokens: int = 900) -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not set (see .env.example)")
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = requests.post(
                GEMINI_URL,
                headers={"x-goog-api-key": key},
                json={
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "systemInstruction": {"parts": [{"text": system}]},
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": max_tokens},
                },
                timeout=60,
            )
            if r.status_code in (429, 500, 503) and attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except (requests.exceptions.RequestException, KeyError, IndexError) as e:
            last = e
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Gemini call failed: {last}")


def format_context(hits: List[dict]) -> str:
    blocks = []
    for n, h in enumerate(hits, 1):
        pages = h["page_start"] if h["page_start"] == h["page_end"] else f"{h['page_start']}-{h['page_end']}"
        blocks.append(f"Excerpt {n} | {h.get('label', '')} | page {pages}\n{h['text']}")
    return "\n\n".join(blocks)


class Pipeline:
    """Base class: subclasses implement `retrieve`; answering is shared."""

    name = ""
    instructions = ""

    def __init__(self, index: Index):
        self.index = index

    @classmethod
    def load(cls) -> "Pipeline":
        return cls(Index.load(cls.name))

    def retrieve(self, question: str) -> List[dict]:
        raise NotImplementedError

    def answer(self, question: str) -> tuple:
        hits = self.retrieve(question)
        prompt = f"Context excerpts:\n\n{format_context(hits)}\n\nQuestion: {question}"
        return call_gemini(f"{BASE_RULES}\n{self.instructions}", prompt), hits
