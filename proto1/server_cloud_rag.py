import os, json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ENV VARS
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "my_docs")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY env vars.")

# Embedder + Qdrant
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)

# FastAPI app
app = FastAPI()

# Allow all origins (for frontend on Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    k: int = 4

@app.post("/chat")
def chat(q: Query):
    # Embed question
    q_emb = embedder.encode([q.question])[0].tolist()

    # Search Qdrant
    hits = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=q_emb,
        limit=q.k
    )

    excerpts = []
    sources = []

    for h in hits:
        p = h.payload

        # Handle dict payload
        if isinstance(p, dict) and "text" in p:
            text = p["text"]
            meta = {k: v for k, v in p.items() if k != "text"}

        # Handle list payload
        elif isinstance(p, list) and len(p) > 0 and isinstance(p[0], dict):
            text = p[0].get("text", "")
            meta = {k: v for k, v in p[0].items() if k != "text"}

        # Fallback
        else:
            text = str(p)
            meta = {}

        excerpts.append(text)
        sources.append(meta)

    # No LLM here → return retrieved excerpts directly
    return {
        "answer": None,
        "retrieved_excerpts": excerpts,
        "sources": sources
    }
