# server_cloud_rag.py
import os, json, requests
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "my_docs")
HF_MODEL = os.environ.get("HF_MODEL")       # optional: e.g. "gpt2" or a small chat model
HF_API_KEY = os.environ.get("HF_API_KEY")   # optional (if using HF inference)
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY env vars.")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)

app = FastAPI()

class Query(BaseModel):
    question: str
    k: int = 4

def hf_generate(prompt, max_tokens=256):
    if not HF_API_KEY or not HF_MODEL:
        return None
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": 0.0}}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    out = r.json()
    # handle typical HF output
    if isinstance(out, list) and "generated_text" in out[0]:
        return out[0]["generated_text"]
    # fallback: stringify response
    return json.dumps(out)

@app.post("/chat")
def chat(q: Query):
    q_emb = embedder.encode([q.question])[0].tolist()
    hits = qdrant.search(collection_name=QDRANT_COLLECTION, query_vector=q_emb, limit=q.k)
    excerpts = []
    sources = []
    for h in hits:
        p = h.payload
        # payload may be dict or list; normalize
        if isinstance(p, dict) and "text" in p:
            text = p["text"]
            meta = {k: v for k, v in p.items() if k != "text"}
        elif isinstance(p, list) and len(p) > 0 and isinstance(p[0], dict):
            first = p[0]
            text = first.get("text", "")
            meta = {k: v for k, v in first.items() if k != "text"}
        else:
            # unknown payload format
            text = str(p)
            meta = {}
        excerpts.append(text)
        sources.append(meta)

    # If you have HF configured, ask HF to generate. Otherwise return excerpts directly.
    prompt = "You are a helpful assistant. Use only these document excerpts to answer the question. If the answer is not present, say 'I don't know'.\n\n"
    for i, e in enumerate(excerpts):
        prompt += f"Excerpt {i+1}:\n{e}\n\n"
    prompt += f"Question: {q.question}\nAnswer:"

    generated = hf_generate(prompt, max_tokens=256)
    if generated:
        return {"answer": generated, "sources": sources}
    else:
        return {"answer": None, "retrieved_excerpts": excerpts, "sources": sources}
