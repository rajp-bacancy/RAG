# RAG Training Projects

This repo holds two separate RAG projects that share one virtual environment and one `.env`:

| Project | What it is | Run with |
|---|---|---|
| **1. What Can I Cook?** (below) | Recipe chatbot: tell it your ingredients, it suggests recipes | `streamlit run app.py` |
| **2. Advanced RAG challenge** ([jump](#advanced-rag-challenge-3-documents-3-strategies)) | Question answering over 3 PDFs, with a different strategy per document | `streamlit run advanced_app.py` |

---

# What Can I Cook?

A simple RAG chatbot: tell it what ingredients you have, and it suggests recipes you can make.

It retrieves matching recipes from a local recipe database (via `sentence-transformers` embeddings) and uses Google's Gemini API to turn them into a friendly, conversational answer.

## Setup

### 1. Create a virtual environment and install dependencies

```bash
cd RAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get a free Gemini API key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey) and sign in with a Google account.
2. Click "Create API key" (no credit card required).

### 3. Configure your API key

```bash
cp .env.example .env
```

Then open `.env` and paste your key:

```
GOOGLE_API_KEY=your-actual-key-here
```

### 4. Build the recipe vector store

```bash
python ingest.py
```

This reads `data/recipes.json` and creates `vector_store.npz` (the searchable recipe index). Re-run this any time you edit `data/recipes.json`.

### 5. Run the app

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) — open it in your browser and start chatting.

## Project structure

| File | Purpose |
|---|---|
| `data/recipes.json` | Recipe dataset (source of truth) |
| `ingest.py` | Builds `vector_store.npz` from `data/recipes.json` |
| `rag.py` | Retrieval + Gemini generation logic |
| `app.py` | Streamlit chat UI |
| `vector_store.npz` | Generated recipe index (not committed to git) |


---

# Advanced RAG challenge (3 documents, 3 strategies)

## How it works (short version)

1. Each PDF is read and cut into small pieces called chunks.
2. Each chunk is stored with its page number and a label saying where it came from.
3. When you ask a question, the system finds the 5-8 chunks that best match it.
4. Only those chunks go to Gemini, which answers from them and cites the pages. If the answer isn't in the chunks, it says so.

## What makes it "advanced"

A basic RAG uses one approach for every document. Here each document gets its own, because the three PDFs have different shapes:

- **GPO (plain text):** search by meaning and by exact keywords together, so exact terms like "$375 million" are found.
- **World Bank (200 pages):** every chunk carries its chapter and section label, and the question is routed to the right section first, so regions and chapters don't get mixed up.
- **iMac (charts, tables, footnotes):** charts are read by Gemini vision, tables become searchable rows, and footnotes are attached to the claims they qualify.

Smaller extras: broad questions such as "summarize this" use a different retrieval path, and the model is told to answer only from the excerpts.

## Details

PDFs live in `data/`. Each use case is a separate pipeline in `advanced_rag/` because the documents have different shapes:

| Use case | Document | What differs | Strategy |
|---|---|---|---|
| 1 | `GPO-Anual-report.pdf` (narrative, 82 pp) | Baseline | Remove running headers/footers, paragraph-aware ~120-word chunks with overlap and nearest-heading prefix, hybrid dense + BM25 retrieval (exact tokens like "$375 million", "FDsys" matter), financial-statement pages kept as verbatim layout tables |
| 2 | `GEP-Jun-2026.pdf` (200 pp, chapters/sections) | Structure | Section tree parsed from the printed table of contents (chapter > section > subsection, boxes, annexes); printed page labels mapped to PDF pages; chunks never cross a section and carry a breadcrumb ("Chapter 2 > South Asia > Outlook") in both the embedding and the LLM context; route-then-fill retrieval: explicit routing (chapter number, executive summary, region name) or section-level retrieval first, then hybrid chunk search inside those sections, plus neighbour chunks and a few global hits as a safety net; forecast tables kept in layout form |
| 3 | `iMac_PER_Oct2024.pdf` (12 pp, charts/tables/footnotes) | Non-text facts | Typed chunks: text with superscript footnote markers recovered as `[^n]`, each endnote as its own chunk (and attached to every chunk citing it), tables rebuilt as markdown plus one sentence per row, charts/icons/big numbers described by Gemini vision from the rendered page (cached in `indexes/imac_figures.json`); numeric questions favour table/figure chunks |

Setup (in addition to the steps above): `pip install -r requirements.txt` and poppler (`pdftotext`, `pdftoppm`; `sudo apt install poppler-utils`).

```bash
python ingest_advanced.py                 # builds indexes/ for gpo, gep, imac (imac calls Gemini once for figures)
python ask_advanced.py gep "What is the growth outlook for South Asia in this report?"
python ask_advanced.py gep "..." --retrieval-only   # show routing + retrieved chunks, no LLM call
python eval_advanced.py --answer          # the 9 questions from the brief, retrieval + answer checks
streamlit run advanced_app.py             # UI with a document picker and retrieved-context viewer
```
