import json

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/recipes.json"
STORE_PATH = "vector_store.npz"


def build_document(recipe: dict) -> str:
    return f"{recipe['name']}. Ingredients: {', '.join(recipe['ingredients'])}."


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        recipes = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    documents = [build_document(r) for r in recipes]
    embeddings = model.encode(documents, normalize_embeddings=True)

    np.savez(
        STORE_PATH,
        embeddings=embeddings,
        recipes=json.dumps(recipes),
    )

    print(f"Ingested {len(recipes)} recipes into '{STORE_PATH}'.")


if __name__ == "__main__":
    main()
