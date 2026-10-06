"""Use case 1 - GPO 2018 Annual Report (plain narrative text).

Baseline pipeline: clean -> paragraph-aware chunking with overlap -> heading-prefixed
embeddings -> hybrid (dense + BM25) retrieval -> grounded answer. BM25 matters here because
questions hinge on exact tokens ("depository libraries", "FDsys", "$375 million").
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple

from .common import DATA_DIR, Index, Pipeline, normalize, pdf_pages, words

PDF = DATA_DIR / "GPO-Anual-report.pdf"
DOC = "GPO 2018 Annual Report"
TARGET_WORDS, MAX_WORDS = 120, 180


def _boilerplate(pages: List[str]) -> set:
    """Lines repeated on many pages (running headers/footers), digits ignored."""
    seen: Counter = Counter()
    for p in pages:
        seen.update({re.sub(r"\d+", "#", l.strip()) for l in p.splitlines() if l.strip()})
    return {l for l, n in seen.items() if n >= max(5, len(pages) // 8)}


def _is_table_page(layout_text: str) -> bool:
    lines = [l for l in layout_text.splitlines() if l.strip()]
    numeric = sum(1 for l in lines if len(re.findall(r"\(?\$?\d[\d,]*\.?\d*\)?", l)) >= 2 and "  " in l)
    prose = sum(1 for l in lines if words(l) >= 12 and "...." not in l)
    return len(lines) > 8 and numeric / len(lines) > 0.4 and prose < 4 and "...." not in layout_text


def _paragraphs(text: str, junk: set) -> List[Tuple[str, bool]]:
    """(paragraph, is_heading). Raw pdftotext wraps lines; blank lines separate blocks."""
    out = []
    for block in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        lines = [l for l in lines if re.sub(r"\d+", "#", l) not in junk]
        if not lines:
            continue
        para = re.sub(r"\s+", " ", " ".join(lines))
        if re.search(r"(?:\b\w ){5,}", para) or re.search(r"www\.gpo\.gov|facebook\.com/USGPO", para):
            continue  # letter-spaced taglines / footer links
        heading = (2 <= words(para) <= 10 and not para.endswith((".", ",", ";")) and not para[0].islower()
                   and not re.search(r"\d", para) and not re.match(r"(Followers|Posts) on", para))
        out.append((para, heading))
    return out


def _sentences(par: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+(?=[A-Z$\d])", par)


def build_chunks() -> List[dict]:
    raw = [normalize(p) for p in pdf_pages(PDF)]
    layout = [normalize(p) for p in pdf_pages(PDF, layout=True)]
    junk = _boilerplate(raw)
    # Short blocks repeated across the report (infographic labels, running titles) are not section headings.
    repeats = Counter(" ".join(b.split()) for r in raw for b in re.split(r"\n\s*\n", r) if words(b) <= 10)
    chunks: List[dict] = []

    paras: List[Tuple[str, int, str]] = []  # (text, page, heading)
    heading = ""
    for pno, (r, lay) in enumerate(zip(raw, layout), 1):
        if _is_table_page(lay):
            # Financial statements: column alignment is the meaning, so keep layout verbatim.
            lines = [l.rstrip() for l in lay.splitlines() if l.strip() and re.sub(r"\d+", "#", l.strip()) not in junk]
            title = " / ".join(l.strip() for l in lines[:3])
            for i in range(0, len(lines), 40):
                body = "\n".join(lines[i : i + 40])
                chunks.append(dict(text=body, embed_text=f"{DOC} - table: {title}\n{body}", label=f"table: {title[:60]}",
                                   page_start=pno, page_end=pno, kind="table"))
            continue
        for para, is_head in _paragraphs(r, junk):
            if is_head and repeats[para] < 3:
                heading = para
            elif is_head:
                continue
            else:
                paras.append((para, pno, heading))

    # Greedy pack paragraphs into ~TARGET_WORDS chunks, overlapping by the last paragraph.
    units: List[Tuple[str, int, str]] = []
    for text, pno, head in paras:
        if words(text) <= MAX_WORDS:
            units.append((text, pno, head))
        else:
            buf = ""
            for s in _sentences(text):
                if buf and words(buf + " " + s) > TARGET_WORDS:
                    units.append((buf, pno, head))
                    buf = s
                else:
                    buf = (buf + " " + s).strip()
            if buf:
                units.append((buf, pno, head))

    cur: List[Tuple[str, int, str]] = []

    def flush():
        if not cur:
            return
        body = "\n\n".join(u[0] for u in cur)
        head = cur[-1][2] or cur[0][2]
        chunks.append(dict(text=body, embed_text=f"{DOC} - {head}\n{body}" if head else f"{DOC}\n{body}",
                           label=head or DOC, page_start=cur[0][1], page_end=cur[-1][1], kind="text"))

    for u in units:
        if cur and sum(words(x[0]) for x in cur) + words(u[0]) > TARGET_WORDS:
            flush()
            last = cur[-1]
            cur = [last] if words(last[0]) < 80 else []
        cur.append(u)
    flush()
    return [c for c in chunks if c["kind"] == "table" or words(c["text"]) >= 30]


def build() -> Index:
    return Index.build(build_chunks())


class GPO(Pipeline):
    name = "gpo"
    instructions = (
        "The document is the U.S. Government Publishing Office (GPO) 2018 Annual Report; "
        "'FY 2018' is the fiscal year ending September 30, 2018. For broad questions (insights, summary, "
        "overview) synthesize the key highlights and figures from the excerpts instead of saying nothing was found."
    )

    OVERVIEW = re.compile(r"\b(insights?|summ?a?r(y|ies|ize|ise)|overview|highlights?|key (points?|takeaways?|findings?)|main (points?|ideas?)|takeaways?|about this (annual )?report|what is this (report|document))\b", re.I)
    OVERVIEW_QUERY = "GPO fiscal year 2018 accomplishments, results, highlights and key figures"

    def retrieve(self, question: str) -> List[dict]:
        if self.OVERVIEW.search(question):
            return self.overview_hits()
        return self.index.search(question, k=6)

    def overview_hits(self, k: int = 8) -> List[dict]:
        """Broad questions match no single passage, so sample the report's narrative part instead:
        best passages for a generic 'highlights' query, one per page, from before the financial statements."""
        narrative_end = min((c["page_start"] for c in self.index.chunks if c["kind"] == "table"), default=40)
        hits = self.index.search(self.OVERVIEW_QUERY, k=60, allow=lambda c: c["kind"] == "text" and c["page_start"] <= min(narrative_end, 40))
        out, pages = [], set()
        for h in hits:
            if h["page_start"] not in pages:
                pages.add(h["page_start"])
                out.append(h)
            if len(out) == k:
                break
        out.sort(key=lambda h: h["page_start"])
        for h in out:
            h["route"] = "overview question: sampled highlights across the narrative pages"
        return out
