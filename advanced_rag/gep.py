"""Use case 2 - World Bank Global Economic Prospects, June 2026 (200 pages, structured).

Design (hierarchical, section-aware):
  1. Parse the printed table of contents into a section tree (chapter > section > subsection),
     and map printed page labels (xvii, 59, ...) to PDF pages.
  2. Locate each TOC heading in the page text, so every line of the report belongs to exactly one
     section. Chunks never cross a section boundary and carry a breadcrumb such as
     "Chapter 2: Regional Outlooks > South Asia > Outlook" that is embedded and shown to the LLM.
  3. Retrieval = route, then fill:
       a. explicit routing (chapter number, "executive summary", region/section names in the question)
       b. otherwise section-level dense retrieval over section summaries
       c. hybrid chunk search restricted to the routed sections, plus a few global hits as a safety net
       d. neighbouring chunks of the best hit are added so answers aren't cut mid-argument.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .common import DATA_DIR, Index, Pipeline, embed, normalize, pdf_pages, words

PDF = DATA_DIR / "GEP-Jun-2026.pdf"
DOC = "Global Economic Prospects, June 2026"
TARGET_WORDS, MAX_WORDS = 170, 230

ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50}
FRONT = ["Acknowledgments", "Foreword", "Executive Summary", "Abbreviations"]
BACK = ["Statistical Appendix", "Data and Forecast Conventions", "Selected Topics"]
CHAPTER_TITLES = {
    1: "Global Outlook",
    2: "Regional Outlooks",
    3: "A Rising Challenge: Sovereign Debt Levels and Interest Rates in EMDEs",
    4: "Navigating Volatility: Fiscal Policy and Commodity Price Swings",
}


def roman(s: str) -> int:
    total = 0
    for i, ch in enumerate(s):
        v = ROMAN[ch]
        total += -v if i + 1 < len(s) and ROMAN[s[i + 1]] > v else v
    return total


def page_label_to_int(label: str) -> Tuple[str, int]:
    return ("r", roman(label)) if not label.isdigit() else ("a", int(label))


# ------------------------------------------------------------------ page labels

def label_offsets(raw: List[str]) -> Tuple[int, int]:
    """(arabic_offset, roman_offset): pdf_index = label + offset. Found by voting on page headers/footers."""
    ara: Dict[int, int] = {}
    rom: Dict[int, int] = {}
    for i, p in enumerate(raw):
        L = [l.strip() for l in p.splitlines() if l.strip()]
        if not L:
            continue
        cands = []
        if L[0].isdigit() and len(L) > 1 and L[1].startswith("CHAPTER"):
            cands.append(L[0])
        if len(L) > 1 and re.fullmatch(r"[0-9ivx]+", L[-1]):
            cands.append(L[-1])
        for c in cands:
            if c.isdigit() and int(c) < 400:
                ara[i - int(c)] = ara.get(i - int(c), 0) + 1
            elif re.fullmatch(r"[ivx]+", c):
                rom[i - roman(c)] = rom.get(i - roman(c), 0) + 1
    return max(ara, key=ara.get), max(rom, key=rom.get)


def pdf_index(label: str, ara_off: int, rom_off: int) -> int:
    kind, n = page_label_to_int(label)
    return n + (ara_off if kind == "a" else rom_off)


def printed_label(idx: int, ara_off: int, rom_off: int, first_arabic_idx: int) -> str:
    if idx >= first_arabic_idx:
        return str(idx - ara_off)
    n = idx - rom_off
    if n <= 0:
        return "cover"
    out, rest = "", n
    for sym, val in (("x", 10), ("ix", 9), ("v", 5), ("iv", 4), ("i", 1)):
        while rest >= val:
            out += sym
            rest -= val
    return out


# ------------------------------------------------------------------ TOC -> section tree

TOC_LINE = re.compile(r"^(?P<title>\S.*?)\s*\.{3,}\s*(?P<page>[0-9]+|[ivx]+)\s*$")
FRONT_BACK = tuple(FRONT + BACK)


def parse_toc(layout: List[str]) -> List[dict]:
    """Entries: dict(level, title, page_label, chapter), read from the detailed 'Contents' pages.

    The layout text keeps indentation, which encodes the hierarchy; a 'Chapter N' prefix is only a
    running header on continuation pages, so the title column (not the line start) gives the depth.
    """
    start = next(i for i, p in enumerate(layout[:15]) if p.lstrip().startswith("Contents"))
    rows = []  # (title, page, col, chapter_prefix)
    pending = None
    done = False
    for p in layout[start : start + 4]:
        if done:
            break
        page_rows = []
        for line in p.splitlines():
            if not line.strip():
                continue
            if line.lstrip().startswith("Annexes"):
                done = True
                break
            m = re.match(r"\s*Chapter\s+(\d)\s{2,}", line)
            prefix = int(m.group(1)) if m else None
            body = line[m.end() :] if m else line.lstrip()
            col = len(line) - len(body)
            t = TOC_LINE.match(body.strip())
            if not t:
                if pending is None and body.strip() not in ("Contents",) and not re.match(r"[ivx]+$", body.strip()):
                    pending = (body.strip(), col, prefix)
                continue
            title = t["title"].strip()
            if pending and abs(pending[1] - col) <= 1:
                title = pending[0] + " " + title
            pending = None
            page_rows.append((title, t["page"], col, prefix))
        sub_cols = [c for t, _, c, _ in page_rows if t not in FRONT_BACK and t not in CHAPTER_TITLES.values()]
        base = min(sub_cols) if sub_cols else 0
        rows.extend((t, pg, c, pre, base) for t, pg, c, pre in page_rows)

    entries: List[dict] = []
    chapter = 0
    for title, page, col, prefix, base in rows:
        if title in FRONT_BACK:
            entries.append(dict(level=1, title=title, page_label=page, chapter=0))
            continue
        ch = next((n for n, t in CHAPTER_TITLES.items() if t == title), None)
        if ch:
            chapter = ch
            entries.append(dict(level=1, title=title, page_label=page, chapter=ch))
            continue
        if prefix:
            chapter = prefix
        entries.append(dict(level=2 + max(0, (col - base) // 3), title=title, page_label=page, chapter=chapter))
    return entries


def add_boxes(entries: List[dict], layout: List[str]) -> List[dict]:
    """Boxes and annexes are listed apart from the section tree (summary page / 'Annexes' block)."""
    summary = next(p for p in layout[:15] if p.lstrip().startswith("Summary of Contents"))
    for m in re.finditer(r"(Box (\d)\.(\d) [^\n]*?)\s*\.{3,}\s*(\d+)", summary):
        entries.append(dict(level=2, title=m.group(1).strip(), page_label=m.group(4), chapter=int(m.group(2))))
    page = next(p for p in layout[:15] if "Annexes" in p and "Contents" not in p[:20].replace("Summary of Contents", ""))
    block = page[page.index("Annexes") : page.index("Boxes")]
    for m in re.finditer(r"(\d)\.(\d)\s{2,}(.*?)\s*\.{3,}\s*(\d+)", block):
        entries.append(dict(level=2, title=f"Annex {m.group(1)}.{m.group(2)} {m.group(3).strip()}",
                            page_label=m.group(4), chapter=int(m.group(1))))
    return entries


# ------------------------------------------------------------------ build

def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def build_chunks() -> Tuple[List[dict], List[dict]]:
    raw = [normalize(p) for p in pdf_pages(PDF)]
    layout = [normalize(p) for p in pdf_pages(PDF, layout=True)]
    ara_off, rom_off = label_offsets(raw)
    first_arabic = 1 + ara_off
    label = lambda i: printed_label(i, ara_off, rom_off, first_arabic)  # noqa: E731

    entries = parse_toc(layout)
    entries = add_boxes(entries, layout)
    for e in entries:
        e["pdf"] = pdf_index(e["page_label"], ara_off, rom_off)
    # Document order: chapters/sections by page; stable for same page (TOC order), boxes slotted by page.
    entries = [e for _, e in sorted(enumerate(entries), key=lambda t: (t[1]["pdf"], t[0]))]

    # Clean line stream: (pdf_idx, line)
    stream: List[Tuple[int, str]] = []
    for i, p in enumerate(raw):
        lines = [l.strip() for l in p.splitlines()]
        keep = []
        for j, l in enumerate(lines):
            if not l:
                keep.append("")
                continue
            if l in ("GLOBAL ECONOMIC PROSPECTS | JUNE 2026",) or re.fullmatch(r"CHAPTER \d", l):
                continue
            if re.fullmatch(r"[0-9]{1,3}|[ivx]+", l) and (j < 3 or j >= len(lines) - 3):
                continue  # page number in header/footer
            keep.append(l)
        stream.extend((i, l) for l in keep)

    # Locate heading positions in the stream, moving forward only.
    pos = 0
    starts: List[Tuple[int, dict]] = []
    for e in entries:
        key = norm_key(e["title"])
        found = None
        for k in range(pos, len(stream)):
            pi, line = stream[k]
            if pi < e["pdf"]:
                continue
            if pi > e["pdf"] + 1:
                break
            lk = norm_key(line)
            if lk and len(lk) >= 4 and (lk == key or (len(lk) >= 12 and key.startswith(lk)) or
                                          (e["title"].lower().startswith(("box", "annex")) and lk.startswith(key[:12]))):
                found = k
                break
        if found is None:  # fall back to the top of the heading's page
            found = next((k for k in range(pos, len(stream)) if stream[k][0] >= e["pdf"]), pos)
        starts.append((found, e))
        pos = found
    starts.sort(key=lambda t: t[0])

    # Breadcrumb for each entry from a level stack.
    def crumb_for(stack: List[dict]) -> str:
        parts = []
        for s in stack:
            if s["level"] == 1 and s["chapter"]:
                parts.append(f"Chapter {s['chapter']}: {s['title']}")
            else:
                parts.append(s["title"])
        return " > ".join(parts)

    sections: List[dict] = []
    stack: List[dict] = []
    for n, (k, e) in enumerate(starts):
        if e["title"].startswith(("Box", "Annex")):
            # Boxes/annexes interrupt the running text; they get their own breadcrumb without touching the stack.
            trail = [s for s in stack if s["level"] == 1] + [e]
        else:
            while stack and stack[-1]["level"] >= e["level"]:
                stack.pop()
            stack.append(e)
            trail = stack
        end = starts[n + 1][0] if n + 1 < len(starts) else len(stream)
        sections.append(dict(breadcrumb=crumb_for(trail), chapter=e["chapter"], title=e["title"],
                             top=trail[0]["title"], start=k, end=end))

    chunks: List[dict] = []
    for si, sec in enumerate(sections):
        lines = stream[sec["start"] : sec["end"]]
        # blank lines delimit paragraphs; wrapped lines joined
        paras: List[Tuple[str, int]] = []
        buf: List[str] = []
        bp = None
        for pi, l in lines + [(lines[-1][0] if lines else 0, "")]:
            if l:
                buf.append(l)
                bp = pi if bp is None else bp
            elif buf:
                paras.append((" ".join(buf), bp))
                buf, bp = [], None
        # Table pages: keep layout verbatim.
        table_pages = {pi for pi, l in lines if l.startswith("TABLE ")}
        cur: List[Tuple[str, int]] = []
        n_chunk = 0

        def flush():
            nonlocal cur, n_chunk
            if not cur:
                return
            body = " ".join(c[0] for c in cur)
            chunks.append(dict(
                text=body, embed_text=f"{sec['breadcrumb']}\n{body}", label=sec["breadcrumb"], section=si,
                chapter=sec["chapter"], top=sec["top"], seq=n_chunk,
                page_start=label(cur[0][1]), page_end=label(cur[-1][1]), kind="text"))
            n_chunk += 1
            cur = []

        for text, pi in paras:
            if pi in table_pages and text.startswith("TABLE "):
                flush()
                cap = text[:90]
                layout_lines = [l.rstrip() for l in layout[pi].splitlines() if l.strip()
                                and "GLOBAL ECONOMIC PROSPECTS" not in l]
                body = "\n".join(layout_lines[:60])
                chunks.append(dict(text=body, embed_text=f"{sec['breadcrumb']}\n{cap}\n{body}",
                                   label=f"{sec['breadcrumb']} > {cap}", section=si, chapter=sec["chapter"],
                                   top=sec["top"], seq=n_chunk, page_start=label(pi), page_end=label(pi), kind="table"))
                n_chunk += 1
                continue
            if pi in table_pages and pi in {c[1] for c in cur} | {pi} and re.match(r"^\d\. ", text):
                continue  # table footnotes already inside the verbatim table chunk
            if words(text) > MAX_WORDS:
                sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
                piece = ""
                for s in sents:
                    if piece and words(piece + " " + s) > TARGET_WORDS:
                        cur.append((piece, pi))
                        flush()
                        piece = s
                    else:
                        piece = (piece + " " + s).strip()
                text = piece
            if cur and sum(words(c[0]) for c in cur) + words(text) > TARGET_WORDS:
                last = cur[-1]
                flush()
                if words(last[0]) < 60:
                    cur = [last]
            cur.append((text, pi))
        flush()
    # Drop tiny/boilerplate chunks
    chunks = [c for c in chunks if words(c["text"]) >= 15 or c["kind"] == "table"]
    for si, sec in enumerate(sections):
        sec["id"] = si
    return chunks, sections


def build() -> Index:
    chunks, all_sections = build_chunks()
    # Section-level records: breadcrumb + opening of the section, used for coarse routing.
    sections = []
    for si, sec in enumerate(all_sections):
        cs = [c for c in chunks if c["section"] == si]
        if not cs:
            continue
        head = " ".join(c["text"] for c in cs[:2])
        head = " ".join(head.split()[:150])
        sections.append(dict(section=si, text=head, embed_text=f"{sec['breadcrumb']}\n{head}", label=sec["breadcrumb"],
                             chapter=sec["chapter"], top=sec["top"], page_start=cs[0]["page_start"],
                             page_end=cs[-1]["page_end"], kind="section"))
    idx = Index.build(chunks)
    sec_emb = embed([s["embed_text"] for s in sections])
    idx.sections = sections
    idx.sec_emb = sec_emb
    return idx


# ------------------------------------------------------------------ retrieval

REGIONS = {
    "east asia and pacific": ["east asia", "eap"],
    "europe and central asia": ["europe and central asia", "eca"],
    "latin america and the caribbean": ["latin america", "caribbean", "lac"],
    "middle east, north africa, afghanistan, and pakistan": ["middle east", "north africa", "mna", "afghanistan", "pakistan"],
    "south asia": ["south asia", "sar"],
    "sub-saharan africa": ["sub-saharan", "sub saharan", "africa", "ssa"],
}


class GEP(Pipeline):
    name = "gep"
    instructions = (
        "The document is the World Bank's Global Economic Prospects, June 2026. Each excerpt is headed by its "
        "chapter > section breadcrumb. Use only excerpts whose breadcrumb matches what the question asks about "
        "(e.g. a specific region or chapter) and never attribute one region's or chapter's figures to another. "
        "State which chapter/section the answer comes from."
    )

    # persisted via Index.save: sections and section embeddings
    @classmethod
    def load(cls) -> "GEP":
        import json
        from .common import INDEX_DIR

        idx = Index.load(cls.name)
        idx.sections = json.loads((INDEX_DIR / "gep_sections.json").read_text())
        idx.sec_emb = np.load(INDEX_DIR / "gep_sections.npy")
        return cls(idx)

    @staticmethod
    def save(idx: Index) -> None:
        import json
        from .common import INDEX_DIR

        idx.save("gep")
        (INDEX_DIR / "gep_sections.json").write_text(json.dumps(idx.sections, ensure_ascii=False))
        np.save(INDEX_DIR / "gep_sections.npy", idx.sec_emb)

    def route(self, question: str) -> Tuple[Optional[set], str]:
        """Return (allowed section ids or None, explanation)."""
        q = question.lower()
        secs = self.index.sections
        # 1. explicit chapter reference
        m = re.search(r"chapter\s+(\d)|\bch\.?\s*(\d)", q)
        chapter = int(m.group(1) or m.group(2)) if m else None
        # 2. executive summary / front matter
        front = next((t for t in FRONT + BACK if t.lower() in q), None)
        # 3. regions
        region = None
        for title, aliases in REGIONS.items():
            if any(re.search(rf"\b{re.escape(a)}\b", q) for a in aliases if a not in ("africa",)):
                region = title
                break
        if region is None and re.search(r"\bafrica\b", q) and "north africa" not in q:
            region = "sub-saharan africa"

        def sec_ids(pred) -> set:
            return {s["section"] for s in secs if pred(s)}

        if front:
            return sec_ids(lambda s: s["top"] == front), f"front/back matter: {front}"
        if region:
            ids = sec_ids(lambda s: s["chapter"] == 2 and norm_key(region) in norm_key(s["label"].split(" > ")[1] if " > " in s["label"] else ""))
            if ids:
                return ids, f"region: {region}"
        if chapter:
            ids = sec_ids(lambda s: s["chapter"] == chapter)
            if ids:
                return ids, f"chapter {chapter}"
        return None, "no explicit routing"

    def retrieve(self, question: str, k: int = 6) -> List[dict]:
        allowed, why = self.route(question)
        route_note = why
        if allowed is None:
            # coarse-to-fine: pick the most relevant sections by embedding, then search inside them
            qv = embed([question])[0]
            sc = self.index.sec_emb @ qv
            top = np.argsort(-sc)[:6]
            allowed = {self.index.sections[i]["section"] for i in top}
            route_note = "section-level retrieval: " + "; ".join(self.index.sections[i]["label"][-50:] for i in top[:3])
        hits = self.index.search(question, k=k, allow=lambda c: c["section"] in allowed)
        # safety net: a few global hits not already included (explicit routing can be wrong)
        extra = [h for h in self.index.search(question, k=3) if h["id"] not in {x["id"] for x in hits}]
        hits = hits + extra[:2 if why.startswith("no") else 1]

        # neighbour expansion for the top hit: previous + next chunk of the same section
        by_key = {(c["section"], c["seq"]): c for c in self.index.chunks}
        if hits:
            best = hits[0]
            have = {h["id"] for h in hits}
            for dseq in (-1, 1):
                nb = by_key.get((best["section"], best["seq"] + dseq))
                if nb and nb["id"] not in have and nb["kind"] == "text":
                    hits.append(dict(nb, score=0.0, dense=0.0, neighbour=True))
        for h in hits:
            h["route"] = route_note
        return hits
