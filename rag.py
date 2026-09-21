from __future__ import annotations

import json
import os
import time

import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

STORE_PATH = "vector_store.npz"
TOP_K = 4

GEMINI_MODEL = "gemini-flash-lite-latest"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)
SYSTEM_INSTRUCTION = (
    "You are 'What Can I Cook?', a friendly conversational cooking assistant. "
    "Chat naturally, like a helpful assistant would. "
    "When the user lists ingredients or asks what they can cook, use the candidate "
    "recipes provided with their message (if any) to suggest the best 1-2 matches: "
    "a one-line reason each fits, any missing ingredients, and a few short steps. "
    "If the user is just greeting you, chatting, asking a follow-up, or saying "
    "thanks, respond naturally and briefly instead of forcing a recipe list. "
    "Keep replies concise and friendly, using markdown when it helps."
)

_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_store = np.load(STORE_PATH, allow_pickle=False)
_embeddings = _store["embeddings"]
_recipes = json.loads(str(_store["recipes"]))
_api_key = os.environ.get("GOOGLE_API_KEY")


def retrieve_recipes(ingredients_text: str, top_k: int = TOP_K) -> list[dict]:
    query_embedding = _embedder.encode([ingredients_text], normalize_embeddings=True)[0]
    scores = _embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [_recipes[i] for i in top_indices]


def build_user_turn(user_text: str, recipes: list[dict]) -> str:
    if not recipes:
        return user_text
    recipes_block = "\n\n".join(
        f"- {r['name']} ({r['cuisine']})\n"
        f"  Ingredients: {', '.join(r['ingredients'])}\n"
        f"  Instructions: {r['instructions']}"
        for r in recipes
    )
    return (
        f"{user_text}\n\n"
        "(For reference, here are recipes from our database that may be relevant "
        f"if this is a cooking request - ignore them if it isn't:\n\n{recipes_block})"
    )


def call_gemini(contents: list[dict], retries: int = 3) -> str:
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.post(
                GEMINI_URL,
                params={"key": _api_key},
                json={
                    "contents": contents,
                    "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
                    "generationConfig": {
                        "temperature": 0.5,
                        "maxOutputTokens": 350,
                    },
                },
                timeout=15,
            )
            if response.status_code in (429, 503) and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise last_error


def get_suggestion(
    user_text: str, history: list[dict] | None = None
) -> tuple[str, list[dict]]:
    recipes = retrieve_recipes(user_text)
    contents = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
        for m in (history or [])
    ]
    contents.append({"role": "user", "parts": [{"text": build_user_turn(user_text, recipes)}]})
    answer = call_gemini(contents)
    return answer, recipes
