# 
import os
import glob
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

#
import PyPDF2
import docx

APP_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(APP_DIR, "docs")
ALLOWED_EXTS = ("*.txt", "*.md", "*.pdf", "*.docx")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K_DEFAULT = 3
INTENT_CONF_THRESHOLD = 0.60  

app = Flask(__name__)
CORS(app)

model = None  # SentenceTransformer instance
documents: List[Dict[str, Any]] = []  # {doc_id, path, text}
chunks: List[Dict[str, Any]] = []  # {chunk_id, doc_id, path, text}
chunk_embeddings: np.ndarray = None

intent_model = None
intent_label_encoder = None
intent_examples = None
intent_responses = None
doc_tfidf_vectorizer = None


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
    if path.lower().endswith(".pdf"):
        return read_pdf(path)
    if path.lower().endswith(".docx"):
        return read_docx(path)
    return read_txt(path)

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    chunk_id = 0
    while start < length:
        end = start + size
        chunk_text = text[start:end]
        chunks.append((chunk_id, start, min(end, length), chunk_text))
        chunk_id += 1
        start = start + size - overlap
    return chunks


def build_index():
    global model, documents, chunks, chunk_embeddings, doc_tfidf_vectorizer

    print("Loading sentence-transformers model:", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    documents = []
    chunks = []
    all_chunk_texts = []
    doc_id = 0
    chunk_global_id = 0

    files = []
    for pat in ALLOWED_EXTS:
        files.extend(glob.glob(os.path.join(DOCS_DIR, pat)))
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

    try:
        texts = [d["text"] for d in documents]
        if texts:
            doc_tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
            doc_tfidf_vectorizer.fit(texts)
    except Exception as e:
        print("Failed to build doc TF-IDF vectorizer:", e)
        doc_tfidf_vectorizer = None


def train_intent_classifier():
    """
    Trains a tiny classifier on example phrases for:
    - greetings
    - how_are_you
    - what_do_you_do
    - what_can_you_answer
    - two_words_about_docs (special)
    - fallback / smalltalk
    """
    global intent_model, intent_label_encoder, intent_examples, intent_responses, model

    # Small example phrases for each intent (feel free to expand)
    intent_examples = {
        "greeting": [
            "hi", "hello", "hey", "hiya", "good morning", "good evening"
        ],
        "how_are_you": [
            "how are you", "how are you doing", "what's up", "how's it going"
        ],
        "what_do_you_do": [
            "what do you do", "what are you", "who are you", "what is your purpose"
        ],
        "what_can_you_answer": [
            "what can you answer", "what can you do", "what are you good at", "what topics can you help with"
        ],
        "two_words_about_docs": [
            "two words about each document", "two words about the documents", "give two words about each doc",
            "brief two-word summary of each document", "two words about docs"
        ],
        "goodbye": [
            "bye", "goodbye", "see you", "see ya", "later"
        ]
    }

    intent_responses = {
        "greeting": "Hi — hello! 👋 How can I help you today?",
        "how_are_you": "I'm an assistant running on your server — ready and functioning!",
        "what_do_you_do": "I search your uploaded documents and answer questions. I can also do short chat replies.",
        "what_can_you_answer": "I can answer factual questions from your docs, give short summaries, and handle simple greetings.",
        "two_words_about_docs": None,  # handled specially (returns structured JSON)
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
        print("Loading embedding model for intent classifier:", EMBEDDING_MODEL)
        temp_model = SentenceTransformer(EMBEDDING_MODEL)
        emb = temp_model.encode(X_texts, convert_to_numpy=True, normalize_embeddings=True)
    else:
        emb = model.encode(X_texts, convert_to_numpy=True, normalize_embeddings=True)

    try:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(emb, y)
        intent_model = clf
        print("Intent classifier trained on", len(X_texts), "examples and", len(intent_label_encoder.classes_), "intents.")
    except Exception as e:
        print("Failed to train intent classifier:", e)
        intent_model = None

def predict_intent(text: str):
    """
    Returns (intent_label:str, confidence:float)
    """
    global intent_model, intent_label_encoder, model
    if intent_model is None or intent_label_encoder is None:
        return None, 0.0
    emb = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    probs = intent_model.predict_proba(emb)[0]  # shape (n_intents,)
    idx = np.argmax(probs)
    label = intent_label_encoder.inverse_transform([idx])[0]
    conf = float(probs[idx])
    return label, conf


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


def get_two_words_per_doc():
    """
    Returns a list of dict: {doc_id, path, two_words: [w1, w2]}
    Uses the fitted doc_tfidf_vectorizer to extract top-scoring terms per document.
    """
    global documents, doc_tfidf_vectorizer
    out = []
    if doc_tfidf_vectorizer is None or not documents:
        for d in documents:
            words = [w for w in (d["text"] or "").split() if len(w) > 2]
            two = words[:2] if words else []
            out.append({"doc_id": d["doc_id"], "path": d["path"], "two_words": two})
        return out

    try:
        X = doc_tfidf_vectorizer.transform([d["text"] for d in documents])  # shape (n_docs, n_features)
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
        # fallback simple approach
        for d in documents:
            words = [w for w in (d["text"] or "").split() if len(w) > 2]
            out.append({"doc_id": d["doc_id"], "path": d["path"], "two_words": words[:2]})
    return out


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
import re

def split_to_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 0]


def synthesize_short_answer(query: str, top_k: int = 3, max_sentences: int = 2):
    """
    Retrieve top_k chunks, split them into sentences, score them by similarity,
    and return up to max_sentences in a conversational form.
    """
    global model, chunks, chunk_embeddings

    results = search_top_k(query, k=top_k)
    if not results:
        return None, []

    combined_text = " ".join([r["text"] for r in results])
    sentences = split_to_sentences(combined_text)

    if not sentences:
        return None, []

    sent_embs = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

    sims = np.dot(sent_embs, q_emb)

    # Pick the top scoring sentences
    top_indices = np.argsort(-sims)[:max_sentences]
    chosen = [sentences[i] for i in sorted(top_indices)]

    # Make conversational answer
    answer = " ".join(s if s.endswith(('.', '?', '!')) else s + '.' for s in chosen)
    answer = "Sure — " + answer

    # Collect sources
    sources = [r["path"] for r in results]

    return answer, sources

@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True)
    q = body.get("question") or body.get("query") or body.get("q")
    top_k = int(body.get("top_k", TOP_K_DEFAULT))
    if not q:
        return jsonify({"error": "no question provided"}), 400

    # 1) predict intent
    intent, conf = predict_intent(q)
    if intent and conf >= INTENT_CONF_THRESHOLD:
        if intent == "two_words_about_docs":
            two_words = get_two_words_per_doc()
            return jsonify({"question": q, "intent": intent, "confidence": conf, "two_words_per_doc": two_words})
        else:
            resp = intent_responses.get(intent, "Sorry, I don't understand.")
            # keep conversational single-string reply
            return jsonify({"question": q, "intent": intent, "confidence": conf, "answer": resp})

    # 2) semantic search + concise synthesis
    if model is None or chunk_embeddings is None:
        return jsonify({"error": "index not built or no docs found"}), 500

    # synthesize up to 2 sentences (conversational)
    convo, sources = synthesize_short_answer(q, top_k=top_k, max_sentences=2)
    if convo:
        # return a single 'answer' string and the source list
        return jsonify({"question": q, "intent": None, "confidence": conf, "answer": convo, "sources": sources})

    # last-resort: short snippet fallback
    results = search_top_k(q, k=top_k)
    reply_texts = []
    for r in results:
        snippet = r["text"]
        if len(snippet) > 200:
            snippet = snippet[:200].rsplit(".", 1)[0] + "..."
        reply_texts.append({"score": r["score"], "text": snippet, "source_path": r["path"]})

    if not reply_texts:
        return jsonify({"question": q, "intent": None, "confidence": conf, "answer": "I couldn't find relevant info in your docs. Try asking differently or 'two words about each document'."})

    # Return the top snippet as a conversational short answer
    top = reply_texts[0]["text"]
    return jsonify({"question": q, "intent": None, "confidence": conf, "answer": top, "sources": [reply_texts[0]["source_path"]]})

# -------------------------
# Startup: build index and train classifier
# -------------------------
if __name__ == "__main__":
    print("Starting backend — building index and training intent classifier...")
    build_index()
    train_intent_classifier()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
