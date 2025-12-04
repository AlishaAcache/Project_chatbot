# app.py
# Finalized: returns only clean answer text (no raw OpenAI objects), robust parsing across SDK shapes,
# retains Qdrant + local fallback behavior, and provides debug/reindex endpoints.
# Based on the project's previous files.

import os
import glob
import time
import uuid
import pathlib
from typing import List, Dict, Any, Optional

import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import PyPDF2
import docx

# OpenAI client
from openai import OpenAI

# Qdrant client (optional)
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_INSTALLED = True
except Exception:
    QDRANT_INSTALLED = False

from PIL import Image
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    TESSERACT_AVAILABLE = False

# -------------------------
# Configuration / env
# -------------------------
APP_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(APP_DIR, "protocols")
os.makedirs(DOCS_DIR, exist_ok=True)

ALLOWED_EXTS = ("*.txt", "*.md", "*.pdf", "*.docx", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "384"))

TOP_K_DEFAULT = int(os.environ.get("TOP_K_DEFAULT", 3))
INTENT_CONF_THRESHOLD = float(os.environ.get("INTENT_CONF_THRESHOLD", 0.35))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "document_chunks")
RECREATE_COLLECTION = os.environ.get("RECREATE_COLLECTION", "false").lower() in ("1", "true", "yes")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set. OpenAI enhancement will be skipped on calls.")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Qdrant client (optional)
qdrant_client = None
if QDRANT_URL and QDRANT_INSTALLED:
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY if QDRANT_API_KEY else None)
        print(f"Connected to Qdrant at {QDRANT_URL}")
    except Exception as e:
        print("Failed to connect to Qdrant:", e)
        qdrant_client = None
else:
    if QDRANT_URL and not QDRANT_INSTALLED:
        print("qdrant-client package missing; QDRANT_URL provided but client not available.")
    else:
        print("QDRANT_URL not set -> running without Qdrant (local in-memory retrieval only).")

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)
CORS(app)

# Globals
model: Optional[SentenceTransformer] = None
documents: List[Dict[str, Any]] = []
chunks: List[Dict[str, Any]] = []
chunk_embeddings: Optional[np.ndarray] = None

intent_model = None
intent_label_encoder = None
intent_examples = None
intent_responses = None
doc_tfidf_vectorizer = None

# -------------------------
# File readers
# -------------------------
def read_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def read_pdf(path: str) -> str:
    text = []
    try:
        reader = PyPDF2.PdfReader(path)
        for page in reader.pages:
            ptext = page.extract_text()
            if ptext:
                text.append(ptext)
    except Exception:
        pass
    return "\n".join(text)

def read_docx(path: str) -> str:
    try:
        doc = docx.Document(path)
        paras = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(paras)
    except Exception:
        return ""

def load_file(path: str) -> str:
    path = str(path)
    lower = path.lower()
    if lower.endswith(".pdf"):
        return read_pdf(path)
    if lower.endswith(".docx"):
        return read_docx(path)
    if lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
        if TESSERACT_AVAILABLE:
            try:
                img = Image.open(path)
                text = pytesseract.image_to_string(img)
                return text or ""
            except Exception:
                return ""
        return ""
    return read_txt(path)

# -------------------------
# Chunking
# -------------------------
def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks_out = []
    start = 0
    length = len(text)
    chunk_id = 0
    while start < length:
        end = start + size
        chunk_text = text[start:end]
        chunks_out.append((chunk_id, start, min(end, length), chunk_text))
        chunk_id += 1
        start = start + size - overlap
    return chunks_out

# -------------------------
# Qdrant helpers
# -------------------------
def ensure_collection():
    if qdrant_client is None:
        return
    try:
        collections = qdrant_client.get_collections()
        names = [c.name for c in collections.collections]
        if RECREATE_COLLECTION and COLLECTION_NAME in names:
            try:
                qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
            except Exception:
                pass
            names = [n for n in names if n != COLLECTION_NAME]
        if COLLECTION_NAME not in names:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )
            print("Created collection:", COLLECTION_NAME)
        else:
            print("Using existing collection:", COLLECTION_NAME)
    except Exception as e:
        print("ensure_collection error:", e)

# -------------------------
# Build local index, optional upload to Qdrant
# -------------------------
def build_index(upload_to_qdrant: bool = True):
    global model, documents, chunks, chunk_embeddings, doc_tfidf_vectorizer

    if model is None:
        print("Loading embedding model:", EMBEDDING_MODEL)
        model = SentenceTransformer(EMBEDDING_MODEL)

    documents = []
    chunks = []
    all_chunk_texts = []
    doc_id = 0
    chunk_global_id = 0

    files = []
    for pat in ALLOWED_EXTS:
        # Recursively search in protocols folder and all subdirectories
        files.extend(glob.glob(os.path.join(DOCS_DIR, "**", pat), recursive=True))
    files = sorted(list(set(files)))

    for path in files:
        text = load_file(path)
        if not text or len(text.strip()) == 0:
            continue
        documents.append({"doc_id": doc_id, "path": path, "text": text})
        c = chunk_text(text)
        for (cid, start, end, ctext) in c:
            chunks.append({
                "chunk_id": chunk_global_id,
                "doc_id": doc_id,
                "path": path,
                "start_char": start,
                "end_char": end,
                "text": ctext.strip()
            })
            all_chunk_texts.append(ctext.strip())
            chunk_global_id += 1
        doc_id += 1

    if len(all_chunk_texts) == 0:
        chunk_embeddings = None
        print("No chunks found when building index.")
    else:
        print(f"Encoding {len(all_chunk_texts)} chunks ...")
        chunk_embeddings = model.encode(all_chunk_texts, show_progress_bar=True, convert_to_numpy=True, batch_size=64, normalize_embeddings=True)
        print("Index built:", len(documents), "docs |", len(chunks), "chunks | embeddings shape:", chunk_embeddings.shape)

    # TF-IDF
    try:
        texts = [d["text"] for d in documents]
        if texts:
            doc_tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
            doc_tfidf_vectorizer.fit(texts)
    except Exception as e:
        print("TFIDF build failed:", e)
        doc_tfidf_vectorizer = None

    # upload to qdrant if configured
    if upload_to_qdrant and qdrant_client is not None and chunk_embeddings is not None:
        ensure_collection()
        try:
            points = []
            for idx, ch in enumerate(chunks):
                emb = chunk_embeddings[idx].tolist()
                p = PointStruct(id=idx, vector=emb, payload={
                    "text": ch["text"],
                    "doc_id": ch["doc_id"],
                    "path": ch["path"],
                    "chunk_id": ch["chunk_id"]
                })
                points.append(p)
            batch = 100
            for i in range(0, len(points), batch):
                qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points[i:i+batch], wait=True)
            print(f"Uploaded {len(points)} points to Qdrant collection {COLLECTION_NAME}")
        except Exception as e:
            print("Failed to upsert points to Qdrant:", e)

# -------------------------
# Intent classifier
# -------------------------
def train_intent_classifier():
    global intent_model, intent_label_encoder, intent_examples, intent_responses, model
    intent_examples = {
        "greeting": ["hi", "hello", "hey", "good morning", "good evening"],
        "how_are_you": ["how are you", "how's it going", "what's up"],
        "what_do_you_do": ["what do you do", "who are you"],
        "two_words_about_docs": ["two words about each document", "two words about the documents"],
        "goodbye": ["bye", "goodbye"]
    }
    intent_responses = {
        "greeting": "Hi — hello! 👋 How can I help you today?",
        "how_are_you": "I'm an assistant running on your server — ready and functioning!",
        "what_do_you_do": "I search your uploaded documents and answer questions.",
        "two_words_about_docs": None,
        "goodbye": "Goodbye — feel free to ask more questions anytime!"
    }
    X_texts = []
    y_labels = []
    for intent, examples in intent_examples.items():
        for ex in examples:
            X_texts.append(ex)
            y_labels.append(intent)
    intent_label_encoder = LabelEncoder()
    y = intent_label_encoder.fit_transform(y_labels)
    if model is None:
        temp_model = SentenceTransformer(EMBEDDING_MODEL)
        emb = temp_model.encode(X_texts, convert_to_numpy=True, normalize_embeddings=True)
    else:
        emb = model.encode(X_texts, convert_to_numpy=True, normalize_embeddings=True)
    try:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(emb, y)
        intent_model = clf
        print("Intent classifier trained.")
    except Exception as e:
        print("Intent training failed:", e)
        intent_model = None

def predict_intent(text: str):
    global intent_model, intent_label_encoder, model
    if intent_model is None or intent_label_encoder is None or model is None:
        return None, 0.0
    try:
        emb = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        probs = intent_model.predict_proba(emb)[0]
        idx = int(np.argmax(probs))
        intent = intent_label_encoder.classes_[idx]
        return intent, float(probs[idx])
    except Exception as e:
        print("predict_intent error:", e)
        return None, 0.0

# -------------------------
# Search helpers
# -------------------------
def search_qdrant_top_k(query: str, k: int = TOP_K_DEFAULT):
    if qdrant_client is None:
        return []
    try:
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).tolist()[0]
        hits = qdrant_client.search(collection_name=COLLECTION_NAME, query_vector=q_emb, limit=k, with_payload=True)
        out = []
        for h in hits:
            out.append({
                "chunk_id": h.payload.get("chunk_id"),
                "doc_id": h.payload.get("doc_id"),
                "path": h.payload.get("path"),
                "text": h.payload.get("text"),
                "score": float(h.score) if h.score is not None else None
            })
        return out
    except Exception as e:
        print("Qdrant search error:", e)
        return []

def search_local_top_k(query: str, k: int = TOP_K_DEFAULT):
    if model is None or chunk_embeddings is None or len(chunks) == 0:
        return []
    try:
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        sims = np.dot(chunk_embeddings, q_emb[0])
        top_idx = np.argsort(-sims)[:k]
        out = []
        for idx in top_idx:
            ch = chunks[int(idx)]
            out.append({
                "chunk_id": ch["chunk_id"],
                "doc_id": ch["doc_id"],
                "path": ch["path"],
                "text": ch["text"],
                "score": float(sims[idx])
            })
        return out
    except Exception as e:
        print("local search error:", e)
        return []

# -------------------------
# OpenAI: safe extract helper (returns str or None)
# -------------------------
def extract_openai_text(resp: Any) -> Optional[str]:
    """
    Robust extractor for various OpenAI SDK response shapes.
    Returns the assistant text content string if found, else None.
    """
    try:
        # Common new-style: choices[0].message["content"]
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            choice0 = resp.choices[0]
            # try dict-like access
            try:
                msg = choice0.message
                # msg may be dict or object
                if isinstance(msg, dict):
                    txt = msg.get("content")
                    if isinstance(txt, str) and txt.strip():
                        return txt.strip()
                else:
                    # object with attributes
                    try:
                        txt = getattr(msg, "content", None)
                        if isinstance(txt, str) and txt.strip():
                            return txt.strip()
                    except Exception:
                        pass
            except Exception:
                pass
            # older style: choices[0].text
            try:
                txt2 = getattr(choice0, "text", None)
                if isinstance(txt2, str) and txt2.strip():
                    return txt2.strip()
            except Exception:
                pass
    except Exception:
        pass
    # Fallbacks failed
    return None

def enhance_with_llm(user_question: str, rag_text: str, sources: List[str]) -> str:
    """
    Returns only the assistant text (string). If no OPENAI_API_KEY, or call fails,
    returns the rag_text (safe fallback).
    """
    if not OPENAI_API_KEY:
        return rag_text

    system_prompt = "You are an assistant that must not invent facts. Use only the provided context."
    user_prompt = f"Context:\n{rag_text}\n\nQuestion: {user_question}\n\nIf you cannot answer from the context, say you could not find the information."

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=800,
            temperature=0.0
        )
        # extract text safely
        text = extract_openai_text(resp)
        if text:
            return text
        # if extraction failed, don't return raw resp; return rag_text
        return rag_text
    except Exception as e:
        print("OpenAI call failed:", e)
        return rag_text

# -------------------------
# Two words per doc
# -------------------------
def get_two_words_per_doc():
    global documents, doc_tfidf_vectorizer
    out = []
    if not documents:
        return out
    if doc_tfidf_vectorizer is None:
        for d in documents:
            words = [w for w in (d["text"] or "").split() if len(w) > 2]
            out.append({"doc_id": d["doc_id"], "path": d["path"], "two_words": words[:2]})
        return out
    try:
        X = doc_tfidf_vectorizer.transform([d["text"] for d in documents])
        feature_names = np.array(doc_tfidf_vectorizer.get_feature_names_out())
        for i, d in enumerate(documents):
            row = X[i].toarray().flatten()
            if np.count_nonzero(row) == 0:
                two = []
            else:
                top_idx = np.argsort(-row)[:2]
                two = [feature_names[j] for j in top_idx if row[j] > 0][:2]
            out.append({"doc_id": d["doc_id"], "path": d["path"], "two_words": two})
    except Exception as e:
        print("two words error:", e)
        for d in documents:
            words = [w for w in (d["text"] or "").split() if len(w) > 2]
            out.append({"doc_id": d["doc_id"], "path": d["path"], "two_words": words[:2]})
    return out

# -------------------------
# HTTP endpoints
# -------------------------
ALLOWED_FILE_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".png", ".jpg", ".jpeg", ".gif", ".webp"}
def allowed_file(filename: str) -> bool:
    return pathlib.Path(filename).suffix.lower() in ALLOWED_FILE_EXTENSIONS

@app.route("/upload", methods=["POST"])
def upload_files():
    try:
        files = request.files.getlist("files")
        question = request.form.get("question", None)
        saved = []
        for f in files:
            filename = secure_filename(f.filename)
            if not filename:
                continue
            if not allowed_file(filename):
                continue
            dst_name = f"{int(time.time())}_{uuid.uuid4().hex}_{filename}"
            dst_path = os.path.join(DOCS_DIR, dst_name)
            f.save(dst_path)
            saved.append(dst_path)
        build_index(upload_to_qdrant=True)
        train_intent_classifier()
        resp = {"ok": True, "saved": [os.path.basename(s) for s in saved]}
        if question:
            resp["message"] = "Files uploaded and index rebuilt. You can now ask your question."
        return jsonify(resp)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/debug_search", methods=["GET"])
def debug_search():
    q = request.args.get("q", "")
    k = int(request.args.get("k", TOP_K_DEFAULT))
    if not q:
        return jsonify({"error": "provide ?q=..."}), 400
    if model is None:
        build_index(upload_to_qdrant=False)
    qdrant_hits = search_qdrant_top_k(q, k) if qdrant_client is not None else []
    local_hits = search_local_top_k(q, k)
    # Return only the minimal info (no raw SDK objects)
    return jsonify({"query": q, "qdrant_hits": qdrant_hits, "local_hits": local_hits})

@app.route("/reindex_qdrant", methods=["POST"])
def reindex_qdrant():
    body = request.get_json(silent=True) or {}
    recreate = bool(body.get("recreate", False))
    try:
        if recreate and qdrant_client is not None:
            try:
                qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
            except Exception:
                pass
        build_index(upload_to_qdrant=True)
        train_intent_classifier()
        return jsonify({"ok": True, "docs": len(documents), "chunks": len(chunks)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json() or {}
    question = payload.get("question") or payload.get("q") or payload.get("text")
    if not question:
        return jsonify({"error": "Missing question"}), 400

    intent, conf = predict_intent(question)
    if intent == "greeting" and conf > INTENT_CONF_THRESHOLD:
        return jsonify({"answer": intent_responses.get("greeting")})
    if intent == "two_words_about_docs" and conf > INTENT_CONF_THRESHOLD:
        two_words = get_two_words_per_doc()
        return jsonify({"two_words_per_doc": two_words})

    if model is None or (chunk_embeddings is None and len(chunks) == 0):
        build_index(upload_to_qdrant=False)
        train_intent_classifier()

    # Try Qdrant then fallback to local
    hits = []
    if qdrant_client is not None:
        hits = search_qdrant_top_k(question, k=TOP_K_DEFAULT)
    if not hits:
        hits = search_local_top_k(question, k=TOP_K_DEFAULT)

    if not hits:
        return jsonify({"answer": "I couldn't find relevant information in your documents. Try reindexing or uploading documents."})

    context_texts = []
    sources = []
    for i, h in enumerate(hits):
        excerpt = (h.get("text") or "").strip()
        if len(excerpt) > 800:
            excerpt = excerpt[:800] + " [...]"
        path = h.get("path") or ""
        context_texts.append(f"Source[{i+1}] ({path.split('/')[-1]}):\n{excerpt}")
        sources.append(path)

    context_block = "\n\n---\n\n".join(context_texts)
    # Generate answer via LLM (returns safe string)
    answer_text = enhance_with_llm(question, context_block, sources)
    # Always return only answer + sources (no raw openai object, no token counts)
    return jsonify({"answer": answer_text, "sources": sources})

@app.route("/doc/<path:filename>", methods=["GET"])
def serve_doc(filename):
    return send_from_directory(DOCS_DIR, filename)

# -------------------------
# Startup
# -------------------------
if __name__ == "__main__":
    try:
        build_index(upload_to_qdrant=True if qdrant_client is not None else False)
        train_intent_classifier()
    except Exception as e:
        print("Initial build failed:", e)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
