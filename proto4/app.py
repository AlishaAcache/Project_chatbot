# app.py
import os
import glob
import json
import numpy as np
import re
import tempfile
from typing import List, Dict, Any, Optional
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# sentence-transformers and ML
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# file readers
import PyPDF2
import docx

# OpenAI client (your original import style)
from openai import OpenAI

# Optional: PIL for image handling
from PIL import Image

# Optional: pytesseract (may not be installed)
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
DOCS_DIR = os.path.join(APP_DIR, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

ALLOWED_EXTS = ("*.txt", "*.md", "*.pdf", "*.docx", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K_DEFAULT = int(os.environ.get("TOP_K_DEFAULT", 3))
INTENT_CONF_THRESHOLD = float(os.environ.get("INTENT_CONF_THRESHOLD", 0.35))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # change if you prefer other model

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set. LLM enhancement will be disabled.")

# Create OpenAI client (works even if key is None; calls will fail if not set)
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)
CORS(app)

# Globals for index and classifier
model = None  # SentenceTransformer instance
documents: List[Dict[str, Any]] = []  # {doc_id, path, text}
chunks: List[Dict[str, Any]] = []  # {chunk_id, doc_id, path, text}
chunk_embeddings: Optional[np.ndarray] = None

# Intent classifier artifacts
intent_model = None
intent_label_encoder = None
intent_examples = None
intent_responses = None
doc_tfidf_vectorizer = None

# -------------------------
# File reading utilities
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
        # Images: try OCR if available, otherwise return empty (image indexing via OCR)
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
# Index builder using sentence-transformers
# -------------------------
def build_index():
    global model, documents, chunks, chunk_embeddings, doc_tfidf_vectorizer

    print("Loading sentence-transformers model:", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    documents = []
    chunks = []
    all_chunk_texts = []
    doc_id = 0
    chunk_global_id = 0

    # find files
    files = []
    for pat in ALLOWED_EXTS:
        files.extend(glob.glob(os.path.join(DOCS_DIR, pat)))
    files = sorted(list(set(files)))

    for path in files:
        text = load_file(path)
        if not text or len(text.strip()) == 0:
            # If image had no OCR text, we still include a placeholder doc with filename for discoverability
            # but skip if truly empty
            # (You may want to extract metadata instead)
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

    # Build a TF-IDF vectorizer for document-level keywords (used to get top-2 terms)
    try:
        texts = [d["text"] for d in documents]
        if texts:
            doc_tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
            doc_tfidf_vectorizer.fit(texts)
    except Exception as e:
        print("Failed to build doc TF-IDF vectorizer:", e)
        doc_tfidf_vectorizer = None

# -------------------------
# Intent classifier
# -------------------------
def train_intent_classifier():
    global intent_model, intent_label_encoder, intent_examples, intent_responses, model

    # Expanded examples for greetings and how_are_you to improve recall
    intent_examples = {
        "greeting": [
            "hi", "hello", "hey", "hiya", "good morning", "good evening",
            "hello there", "hi there", "hey!", "yo", "greetings", "hey there"
        ],
        "how_are_you": [
            "how are you", "how are you doing", "what's up", "how's it going",
            "how are you?", "how r u", "how are you doing today", "how's everything"
        ],
        "what_do_you_do": ["what do you do", "what are you", "who are you", "what is your purpose"],
        "what_can_you_answer": ["what can you answer", "what can you do", "what are you good at", "what topics can you help with"],
        "two_words_about_docs": ["two words about each document", "two words about the documents", "give two words about each doc", "brief two-word summary of each document", "two words about docs"],
        "goodbye": ["bye", "goodbye", "see you", "see ya", "later"]
    }

    intent_responses = {
        "greeting": "Hi — hello! 👋 How can I help you today?",
        "how_are_you": "I'm an assistant running on your server — ready and functioning!",
        "what_do_you_do": "I search your uploaded documents and answer questions. I can also do short chat replies.",
        "what_can_you_answer": "I can answer factual questions from your docs, give short summaries, and handle simple greetings.",
        "two_words_about_docs": None,
        "goodbye": "Goodbye — feel free to ask more questions anytime!"
    }

    # Prepare training data
    X_texts = []
    y_labels = []
    for intent, examples in intent_examples.items():
        for ex in examples:
            X_texts.append(ex)
            y_labels.append(intent)

    # Label encode
    intent_label_encoder = LabelEncoder()
    y = intent_label_encoder.fit_transform(y_labels)

    # Use sentence-transformers embeddings for classifier features
    if model is None:
        print("Loading embedding model for intent classifier:", EMBEDDING_MODEL)
        temp_model = SentenceTransformer(EMBEDDING_MODEL)
        emb = temp_model.encode(X_texts, convert_to_numpy=True, normalize_embeddings=True)
    else:
        emb = model.encode(X_texts, convert_to_numpy=True, normalize_embeddings=True)

    # Train a small logistic regression (probabilistic)
    try:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(emb, y)
        intent_model = clf
        print("Intent classifier trained on", len(X_texts), "examples and", len(intent_label_encoder.classes_), "intents.")
    except Exception as e:
        print("Failed to train intent classifier:", e)
        intent_model = None

def predict_intent(text: str):
    global intent_model, intent_label_encoder, model
    if intent_model is None or intent_label_encoder is None:
        return None, 0.0
    emb = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    probs = intent_model.predict_proba(emb)[0]
    idx = np.argmax(probs)
    label = intent_label_encoder.inverse_transform([idx])[0]
    conf = float(probs[idx])
    return label, conf

# -------------------------
# Search helper
# -------------------------
def search_top_k(query: str, k: int = TOP_K_DEFAULT):
    global model, chunks, chunk_embeddings
    if model is None or chunk_embeddings is None or len(chunks) == 0:
        return []

    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sims = np.dot(chunk_embeddings, q_emb[0])
    topk_idx = np.argsort(-sims)[:k]
    results = []
    for idx in topk_idx:
        score = float(sims[idx])
        ch = chunks[int(idx)]
        results.append({
            "chunk_id": ch["chunk_id"],
            "doc_id": ch["doc_id"],
            "path": ch["path"],
            "text": ch["text"],
            "score": score
        })
    return results

# -------------------------
# Two-word descriptors for documents (using TF-IDF top terms)
# -------------------------
def get_two_words_per_doc():
    global documents, doc_tfidf_vectorizer
    out = []
    if doc_tfidf_vectorizer is None or not documents:
        for d in documents:
            words = [w for w in (d["text"] or "").split() if len(w) > 2]
            two = words[:2] if words else []
            out.append({"doc_id": d["doc_id"], "path": d["path"], "two_words": two})
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
        print("Error getting two words per doc:", e)
        for d in documents:
            words = [w for w in (d["text"] or "").split() if len(w) > 2]
            out.append({"doc_id": d["doc_id"], "path": d["path"], "two_words": words[:2]})
    return out

# -------------------------
# Sentence splitting and synthesis
# -------------------------
def split_to_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 0]

def synthesize_short_answer(query: str, top_k: int = 3, max_sentences: Optional[int] = 2):
    """
    If max_sentences is None -> return the full combined extracted text (no summarization).
    Otherwise, pick the top `max_sentences` sentences from the combined text.
    """
    global model, chunks, chunk_embeddings

    results = search_top_k(query, k=top_k)
    if not results:
        return None, []

    combined_text = " ".join([r["text"] for r in results])
    sources = [r["path"] for r in results]

    if max_sentences is None:
        return combined_text, sources

    sentences = split_to_sentences(combined_text)
    if not sentences:
        return combined_text, sources

    sent_embs = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    sims = np.dot(sent_embs, q_emb)
    top_indices = np.argsort(-sims)[:max_sentences]
    chosen = [sentences[i] for i in sorted(top_indices)]
    answer = " ".join(s if s.endswith(('.', '?', '!')) else s + '.' for s in chosen)
    answer = "Sure — " + answer
    return answer, sources

# -------------------------
# OpenAI LLM helper
# -------------------------
def enhance_with_llm(user_question: str, rag_answer: str, sources: List[str]) -> str:
    """
    Send a constrained prompt to OpenAI to rewrite/improve the RAG answer.
    If OPENAI_API_KEY is not set or call fails, returns the original rag_answer.
    """
    if not OPENAI_API_KEY:
        return rag_answer

    prompt = f"""
You are a helpful assistant that must NOT hallucinate.
User question: {user_question}

RAG extracted answer (use only this info; do not add new facts): 
{rag_answer}

Sources: {', '.join([s.split('/')[-1] for s in sources]) if sources else 'none'}

Task: Rewrite the RAG answer into a clear, friendly final answer. Preserve all details present in the RAG extracted answer and do NOT shorten or summarize it. You may expand clarifications using only information present in the RAG text, and you may reorganize for clarity. If the RAG answer is uncertain, explicitly say "I couldn't find a definite answer in the documents."
"""

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.0
        )
        msg = resp.choices[0].message.content.strip()
        if not msg:
            return rag_answer
        return msg
    except Exception as e:
        print("OpenAI call failed:", e)
        return rag_answer

# -------------------------
# Flask endpoints (chat + index reload + docs summary) - unchanged logically
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return "RAG semantic-search backend (sentence-transformers) with simple intents running"

@app.route("/reload", methods=["GET", "POST"])
def reload_index():
    build_index()
    train_intent_classifier()
    return jsonify({"status": "ok", "docs": len(documents), "chunks": len(chunks)})

@app.route("/doc_summary", methods=["GET"])
def doc_summary():
    out = get_two_words_per_doc()
    return jsonify({"docs": out})

@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True)
    q = body.get("question") or body.get("query") or body.get("q")
    top_k = int(body.get("top_k", TOP_K_DEFAULT))
    if not q:
        return jsonify({"error": "no question provided"}), 400

    # ---- Rule-based quick check for greetings & smalltalk (fast, deterministic) ----
    lower_q = (q or "").strip().lower()
    if lower_q in ("hi", "hello", "hey", "hiya", "yo", "greetings", "hey there", "hi there") \
        or lower_q.startswith("hi ") or lower_q.startswith("hello ") \
        or any(phrase in lower_q for phrase in ["how are you", "how's it going", "how r u", "how are u", "how are you doing"]):
        return jsonify({
            "question": q,
            "intent": "greeting",
            "confidence": 1.0,
            "answer": "Hi — hello! 👋 How can I help you today?"
        })

    # 1) predict intent (ML)
    intent, conf = predict_intent(q)
    print(f"[DEBUG] query='{q}' -> intent={intent}, conf={conf}")
    if intent and conf >= INTENT_CONF_THRESHOLD:
        if intent == "two_words_about_docs":
            two_words = get_two_words_per_doc()
            return jsonify({"question": q, "intent": intent, "confidence": conf, "two_words_per_doc": two_words})
        else:
            resp = intent_responses.get(intent, "Sorry, I don't understand.")
            return jsonify({"question": q, "intent": intent, "confidence": conf, "answer": resp})

    # 2) semantic search + full synthesis (no forced shortening)
    if model is None or chunk_embeddings is None:
        return jsonify({"error": "index not built or no docs found"}), 500

    convo, sources = synthesize_short_answer(q, top_k=top_k, max_sentences=None)
    if convo:
        final_answer = enhance_with_llm(q, convo, sources)
        return jsonify({"question": q, "intent": None, "confidence": conf, "answer": final_answer, "sources": sources})

    # fallback: search snippets
    results = search_top_k(q, k=top_k)
    reply_texts = []
    for r in results:
        snippet = r["text"]
        reply_texts.append({"score": r["score"], "text": snippet, "source_path": r["path"]})

    if not reply_texts:
        return jsonify({"question": q, "intent": None, "confidence": conf, "answer": "I couldn't find relevant info in your docs. Try asking differently or 'two words about each document'."})

    top = reply_texts[0]["text"]
    final_answer = enhance_with_llm(q, top, [reply_texts[0]["source_path"]])
    return jsonify({"question": q, "intent": None, "confidence": conf, "answer": final_answer, "sources": [reply_texts[0]["source_path"]]})

# -------------------------
# New: Upload endpoint (saves files into DOCS_DIR, attempts OCR for images, reindexes)
# -------------------------
ALLOWED_UPLOAD_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.png', '.jpg', '.jpeg', '.gif', '.webp'}

def allowed_upload(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_UPLOAD_EXTENSIONS

def ocr_image_to_text(path: str) -> Optional[str]:
    if not TESSERACT_AVAILABLE:
        return None
    try:
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return text.strip() if text and text.strip() else None
    except Exception as e:
        app.logger.exception("OCR failed: %s", e)
        return None

@app.route("/upload", methods=["POST"])
def upload_files():
    """
    Accepts multipart/form-data:
      - files: one or more files
      - question (optional): string
    Saves files into DOCS_DIR and triggers build_index() so they become searchable.
    Returns JSON with per-file status and optional OCR/extracted text.
    """
    try:
        if 'files' not in request.files and 'file' not in request.files:
            return jsonify({"success": False, "error": "No files provided"}), 400

        files = request.files.getlist('files') or request.files.getlist('file')
        question = request.form.get('question') or request.form.get('prompt') or request.args.get('question') or None

        saved = []
        for f in files:
            if f.filename == "":
                continue
            filename = secure_filename(f.filename)
            if not allowed_upload(filename):
                saved.append({"filename": filename, "ok": False, "error": "extension not allowed"})
                continue
            # Ensure unique name in docs folder
            dest_name = f"{next(tempfile._get_candidate_names())}_{filename}"
            dest_path = os.path.join(DOCS_DIR, dest_name)
            f.save(dest_path)
            file_info = {"filename": filename, "saved_as": dest_path, "ok": True, "ocr_text": None}

            # If image, try OCR (if available)
            if dest_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')) and TESSERACT_AVAILABLE:
                ocr = ocr_image_to_text(dest_path)
                file_info["ocr_text"] = ocr
                # Optionally save OCR text as .txt sidecar so the index picks it up immediately
                if ocr:
                    try:
                        txt_sidecar = dest_path + ".ocr.txt"
                        with open(txt_sidecar, "w", encoding="utf-8") as tf:
                            tf.write(ocr)
                        file_info["ocr_sidecar"] = txt_sidecar
                    except Exception:
                        file_info["ocr_sidecar"] = None

            saved.append(file_info)

        # After saving uploaded files, rebuild the index so chat/search sees them
        build_index()
        train_intent_classifier()

        # If a question was provided, attempt to answer it immediately using /chat logic:
        if question:
            # call synthesize + optional LLM enhancement (reuse functions)
            convo, sources = synthesize_short_answer(question, top_k=TOP_K_DEFAULT, max_sentences=None)
            if convo:
                final_answer = enhance_with_llm(question, convo, sources)
                return jsonify({"success": True, "saved": saved, "answer": final_answer, "sources": sources})
            else:
                return jsonify({"success": True, "saved": saved, "message": "Files uploaded and indexed, but no immediate answer found."})

        return jsonify({"success": True, "saved": saved, "message": "Files uploaded and indexed."})

    except Exception as e:
        app.logger.exception("Upload error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500

# Serve docs for debugging if needed
@app.route("/docs/<path:filename>", methods=["GET"])
def serve_doc(filename):
    return send_from_directory(DOCS_DIR, filename, as_attachment=False)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "tesseract_available": TESSERACT_AVAILABLE,
        "openai_api_key_configured": bool(OPENAI_API_KEY),
        "model_loaded": model is not None
    })

# -------------------------
# Startup
# -------------------------
if __name__ == "__main__":
    print("Starting backend — building index and training intent classifier...")
    build_index()
    train_intent_classifier()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
