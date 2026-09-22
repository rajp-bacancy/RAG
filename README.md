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
