"""Use case 3 - Apple iMac Product Environmental Report (charts, tables, footnotes).

Plain paragraph extraction misses most of what matters here, so the document is split into typed chunks:
  text      body paragraphs, with superscript footnote markers recovered as [^n]
  footnote  each numbered endnote as its own chunk (and attached to every chunk that cites it)
  table     rebuilt from the layout text: a markdown table + one self-describing sentence per row
  figure    chart / icon / big-number descriptions written by a vision model from the rendered page
            (cached in indexes/imac_figures.json so ingest only pays for it once)
Retrieval is hybrid + type-aware: numeric questions favour table/figure chunks, and any retrieved chunk
pulls in the footnotes that qualify it (config used, definitions, caveats).
"""
from __future__ import annotations

import base64
import json
import os
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .common import DATA_DIR, GEMINI_URL, INDEX_DIR, Index, Pipeline, normalize, pdf_pages, words

PDF = DATA_DIR / "iMac_PER_Oct2024.pdf"
DOC = "iMac Product Environmental Report (Apple, Oct 2024)"
FIGURE_CACHE = INDEX_DIR / "imac_figures.json"
FIGURE_PAGES = [1, 2, 3, 4, 6, 7, 10]  # pages whose facts live in graphics, icons or big-number call-outs

SECTIONS = [  # (heading as it appears on the page, section name) - first match wins
    ("Progress toward carbon neutral", "Progress toward carbon neutral"),
    ("Taking responsibility for", "Taking responsibility"),
    ("Design and Source", "Design and Source"),
    ("Package and Ship", "Package and Ship"),
    ("Supplier Code of Conduct sets", "Make"),
    ("Recover\nReturn", "Recover"),
    ("Definitions", "Definitions"),
    ("Endnotes", "Endnotes"),
    ("Carbon \n", "Carbon Footprint"),
    ("iMac uses 58 percent less energy", "Use"),
    ("Product Environmental Report\n", "Overview"),
]

# Row-level semantics for the two data tables (the PDF gives only bare labels and numbers).
CARBON_TITLE = "iMac (two ports) 256GB carbon footprint (greenhouse gas emissions, life cycle assessment)"
POWER_TITLE = "Power consumption for iMac (ENERGY STAR test modes, endnote 18)"


# ------------------------------------------------------------------ superscripts -> [^n]

def superscript_markers(pdf_path: Path) -> Dict[int, List[tuple]]:
    """page -> [(preceding word, footnote number)] using font size (superscripts are ~65% of body size)."""
    import pdfplumber

    out: Dict[int, List[tuple]] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for pno, page in enumerate(pdf.pages, 1):
            ws = page.extract_words(extra_attrs=["size"])
            found = []
            for w in ws:
                if not re.fullmatch(r"\d{1,2}", w["text"]):
                    continue
                # nearest word to the left on (roughly) the same line; extraction order is by `top`, so use geometry
                left = [o for o in ws if o is not w and o["x1"] <= w["x0"] + 2 and o["x0"] < w["x0"]
                        and o["top"] - 3 <= w["bottom"] and o["bottom"] + 6 >= w["top"]]
                if not left:
                    continue
                prev = max(left, key=lambda o: o["x1"])
                if prev["text"].endswith("CO"):
                    continue  # CO2 subscript, not a footnote
                if w["x0"] - prev["x1"] < 4 and w["size"] < prev["size"] * 0.8:
                    found.append((prev["text"], int(w["text"])))
            out[pno] = found
    return out


def mark_footnotes(text: str, markers: List[tuple]) -> str:
    for prev, n in markers:
        pat = re.escape(prev) + r"\s?" + str(n) + r"(?!\d)"
        text, k = re.subn(pat, lambda m: f"{prev}[^{n}]", text, count=1)
    return text


# ------------------------------------------------------------------ endnotes

def parse_endnotes(layout_pages: List[str]) -> Dict[int, dict]:
    """Numbered endnotes from the layout text of the endnote pages. Numbers must run 1,2,3,... in order."""
    notes: Dict[int, dict] = {}
    cur = 0
    for pno in (11, 12):
        for line in layout_pages[pno - 1].splitlines():
            s = line.strip()
            if not s or s.startswith("Endnotes") or s.startswith("iMac |") or s.startswith("©"):
                continue
            m = re.match(r"(\d{1,2})\s*(?=[A-Za-z])(.*)", s)
            if m and int(m.group(1)) == cur + 1:
                cur += 1
                notes[cur] = dict(page=pno, text=m.group(2).strip())
            elif cur:
                cells = split_cells(line)
                if "Power consumption for iMac" in s or s.startswith("Mode") or (len(cells) >= 3 and re.match(r"[\d.]+(W|%)$", cells[-1])):
                    continue  # table rows are captured separately as table chunks
                if s.startswith("©") or s.startswith("Mac, the Mac logo") or "trademarks" in s or "Apple Store is" in s:
                    continue
                notes[cur]["text"] += " " + s
    for n in notes.values():
        n["text"] = re.sub(r"\s+", " ", n["text"]).replace("- ", "-") if False else re.sub(r"\s+", " ", n["text"])
    return notes


# ------------------------------------------------------------------ tables

def split_cells(line: str) -> List[str]:
    return [c.strip() for c in re.split(r"\s{2,}", line.strip()) if c.strip()]


def md_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def power_table(layout_pages: List[str]) -> Optional[dict]:
    lines = layout_pages[11].splitlines()
    rows = []
    for l in lines:
        c = split_cells(l)
        if len(c) == 4 and re.match(r"[\d.]+(W|%)", c[1]):
            rows.append(c)
    if not rows:
        return None
    header = ["Mode", "100V", "115V", "230V"]
    sentences = [
        f"iMac power consumption in '{r[0]}' mode (ENERGY STAR test): {r[1]} at 100V, {r[2]} at 115V, {r[3]} at 230V."
        if r[0] != "Power adapter efficiency"
        else f"iMac 143W power adapter efficiency (average at 100/75/50/25% load): {r[1]} at 100V, {r[2]} at 115V, {r[3]} at 230V."
        for r in rows
    ]
    return dict(page=12, title=POWER_TITLE, markdown=md_table(header, rows), sentences=sentences, notes=[18])


def carbon_table(layout_pages: List[str]) -> Optional[dict]:
    lines = layout_pages[9].splitlines()
    rows = []
    for l in lines:
        c = split_cells(l)
        if len(c) == 2 and re.search(r"(kg CO2e|%|<1%|↓)", c[1]) and not c[0].startswith("Note"):
            rows.append(c)
    if not rows:
        return None
    sentences = []
    phases = {"Production", "Transportation", "Product use", "End-of-life processing"}
    for label, val in rows:
        label = re.sub(r"^[•\s]+", "", label)
        if label in phases:
            sentences.append(f"Share of the iMac (two ports) 256GB life cycle carbon footprint (346 kg CO2e) from {label}: {val}.")
        elif label.startswith("iMac (four ports)"):
            sentences.append(f"Product carbon footprint of the iMac (four ports) 512GB configuration: {val}.")
        elif label.startswith("GHG reductions"):
            sentences.append(f"Greenhouse gas reductions achieved for iMac (two ports) 256GB: {val} (endnote 7).")
        else:
            sentences.append(f"iMac (two ports) 256GB - {label}: {val}.")
    rows_md = [[re.sub(r"^[•\s]+", "", a), b] for a, b in rows]
    return dict(page=10, title=CARBON_TITLE, markdown=md_table(["Greenhouse gas emissions", "iMac (two ports) 256GB / config"], rows_md),
                sentences=sentences, notes=[7])


# ------------------------------------------------------------------ figures (vision)

FIGURE_PROMPT = (
    "This is one page of Apple's iMac Product Environmental Report. List every chart, diagram, icon call-out and "
    "big-number statistic on the page as short factual bullet points. For charts, give each bar/segment label with its "
    "number and unit and what it represents (e.g. baseline, reduction, final). Include exact numbers, percentages and "
    "units. Write superscript footnote markers as [^n]. Skip decoration and ordinary paragraphs. "
    "Do not add anything that is not visible on the page."
)


def describe_pages(pdf_path: Path, pages: List[int]) -> Dict[str, str]:
    cache: Dict[str, str] = json.loads(FIGURE_CACHE.read_text()) if FIGURE_CACHE.exists() else {}
    key = os.environ.get("GOOGLE_API_KEY")
    for p in pages:
        if str(p) in cache or not key:
            continue
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["pdftoppm", "-r", "110", "-png", "-f", str(p), "-l", str(p), str(pdf_path), f"{td}/pg"], check=True)
            png = next(Path(td).glob("pg*.png")).read_bytes()
        body = {"contents": [{"role": "user", "parts": [
            {"text": FIGURE_PROMPT},
            {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(png).decode()}}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 1200}}
        for attempt in range(4):
            r = requests.post(GEMINI_URL, headers={"x-goog-api-key": key}, json=body, timeout=90)
            if r.status_code in (429, 500, 503):
                import time
                time.sleep(4 * (attempt + 1))
                continue
            r.raise_for_status()
            cache[str(p)] = "".join(x.get("text", "") for x in r.json()["candidates"][0]["content"]["parts"]).strip()
            break
        INDEX_DIR.mkdir(exist_ok=True)
        FIGURE_CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    return cache


# ------------------------------------------------------------------ build

def section_of(raw_page: str, pno: int) -> str:
    if pno == 1:
        return "Overview (progress toward 2030 goal, icon call-outs)"
    for needle, name in SECTIONS:
        if needle in raw_page:
            return name
    return "iMac Product Environmental Report"


def refs(text: str) -> List[int]:
    return sorted({int(n) for n in re.findall(r"\[\^(\d{1,2})\]", text)})


_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def clean_description(desc: str) -> str:
    """Drop the model's chatty preamble; turn unicode superscripts (¹⁷) into [^17] markers."""
    lines = desc.splitlines()
    if lines and re.match(r"(Here are|Based on|The following)", lines[0]):
        lines = lines[1:]
    text = "\n".join(lines).replace("**", "")
    return re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+", lambda m: f"[^{m.group(0).translate(_SUP)}]", text).strip()


def build_chunks(with_figures: bool = True) -> List[dict]:
    raw = [normalize(p) for p in pdf_pages(PDF)]
    layout = [normalize(p) for p in pdf_pages(PDF, layout=True)]
    marks = superscript_markers(PDF)
    chunks: List[dict] = []

    def add(kind: str, page: int, section: str, text: str, notes: Optional[List[int]] = None, label: str = ""):
        notes = sorted(set(notes or []) | set(refs(text)))
        chunks.append(dict(text=text, embed_text=f"{DOC} | {section} | {kind}\n{text}", label=label or f"{section} ({kind})",
                           page_start=str(page), page_end=str(page), kind=kind, notes=notes, section=section))

    # 1. body text (not the endnote pages)
    for pno in range(1, 11):
        text = mark_footnotes(raw[pno - 1], marks.get(pno, []))
        section = section_of(raw[pno - 1], pno)
        blocks = [re.sub(r"\s+", " ", b).strip() for b in re.split(r"\n\s*\n", text)]
        blocks = [b for b in blocks if b and not b.startswith("iMac | Product Environmental Report")]
        if pno == 10:
            blocks = [b for b in blocks if not re.search(r"Total product footprint|Life cycle product emissions|Configuration", b)]
        cur: List[str] = []
        for b in blocks + [None]:
            if b is None or (cur and words(" ".join(cur)) + words(b) > 170):
                if cur:
                    add("text", pno, section, " ".join(cur))
                cur = []
            if b:
                cur.append(b)

    # 2. endnotes: one chunk per note
    notes = parse_endnotes(layout)
    for n, d in notes.items():
        add("footnote", d["page"], f"Endnote {n}", f"Endnote {n} (page {d['page']}): {d['text']}", label=f"Endnote {n}")
        chunks[-1]["note_no"] = n

    # 3. tables
    for tbl in (carbon_table(layout), power_table(layout)):
        if not tbl:
            continue
        add("table", tbl["page"], tbl["title"], f"{tbl['title']}\n{tbl['markdown']}", tbl["notes"], label=f"Table: {tbl['title'][:50]}")
        for s in tbl["sentences"]:
            add("table", tbl["page"], tbl["title"], s, tbl["notes"], label=f"Table row: {tbl['title'][:40]}")

    # 4. figures / icons / big numbers (vision-described)
    if with_figures:
        for p, desc in describe_pages(PDF, FIGURE_PAGES).items():
            desc = clean_description(desc)
            add("figure", int(p), f"Graphics on page {p}", f"Graphics and call-outs on page {p}:\n{desc}", label=f"Figure/graphic, page {p}")
    return chunks


def build() -> Index:
    return Index.build(build_chunks())


# ------------------------------------------------------------------ retrieval

NUMERIC = re.compile(r"\b(how much|how many|percent|percentage|%|kg|co2|watt|\bw\b|power|total|footprint|energy|emission|recycled|less)\b", re.I)
FOOTNOTE_Q = re.compile(r"\b(footnote|endnote|note \d+|caveat|defin|how (was|is) .* (calculated|measured)|based on|assum)", re.I)


class IMac(Pipeline):
    name = "imac"
    instructions = (
        "The excerpts mix body text, table rows, descriptions of charts/icons and endnotes ([^n] marks a footnote "
        "reference; matching 'Endnote n' text is attached). Prefer the exact figure from a table or chart over a "
        "general sentence, give the configuration the number applies to, and mention a footnote if it limits the claim."
    )

    def retrieve(self, question: str, k: int = 6) -> List[dict]:
        numeric = bool(NUMERIC.search(question))
        footq = bool(FOOTNOTE_Q.search(question))

        def boost(c: dict) -> float:
            b = 0.0
            if numeric and c["kind"] in ("table", "figure"):
                b += 0.5
            if footq and c["kind"] == "footnote":
                b += 1.5
            return b

        hits = self.index.search(question, k=k, boost=boost)
        have = {h["id"] for h in hits}
        by_note = {c["note_no"]: c for c in self.index.chunks if c["kind"] == "footnote"}
        attached = []
        for h in hits:
            for n in h.get("notes", []):
                fn = by_note.get(n)
                if fn and fn["id"] not in have:
                    have.add(fn["id"])
                    attached.append(dict(fn, score=0.0, dense=0.0, attached_to=h["id"]))
        return hits + attached
