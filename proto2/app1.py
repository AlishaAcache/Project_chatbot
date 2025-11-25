# SIMPLE RAG SHORT-ANSWER BACKEND
#import os
# import glob
# from flask import Flask, request, jsonify
# from flask_cors import CORS

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# from PyPDF2 import PdfReader
# from docx import Document

# # -----------------------------
# # Config
# # -----------------------------
# DOCS_DIR = "docs"
# ALLOWED_EXTS = ("*.txt", "*.md", "*.pdf", "*.docx")
# CHUNK_SIZE = 850
# CHUNK_OVERLAP = 150

# # -----------------------------
# # State
# # -----------------------------
# documents = []
# chunks = []
# vectorizer = None
# chunk_vectors = None

# # -----------------------------
# # Readers
# # -----------------------------
# def read_txt(path):
#     try:
#         with open(path, "r", encoding="utf-8", errors="ignore") as f:
#             return f.read()
#     except:
#         return ""

# def read_pdf(path):
#     try:
#         r = PdfReader(path)
#         pages = []
#         for p in r.pages:
#             try:
#                 t = p.extract_text()
#             except:
#                 t = None
#             if t:
#                 pages.append(t)
#         return "\n".join(pages)
#     except Exception as e:
#         print("PDF read error", path, e)
#         return ""

# def read_docx(path):
#     try:
#         d = Document(path)
#         paras = [p.text for p in d.paragraphs if p.text]
#         return "\n".join(paras)
#     except Exception as e:
#         print("DOCX read error", path, e)
#         return ""

# def load_file(path):
#     lp = path.lower()
#     if lp.endswith(".txt") or lp.endswith(".md"):
#         return read_txt(path)
#     if lp.endswith(".pdf"):
#         return read_pdf(path)
#     if lp.endswith(".docx"):
#         return read_docx(path)
#     return ""

# # -----------------------------
# # Chunking
# # -----------------------------
# def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
#     if not text:
#         return []
#     text = text.replace("\r\n", "\n")
#     out = []
#     start = 0
#     L = len(text)
#     while start < L:
#         end = start + size
#         out.append(text[start:end].strip())
#         if end >= L:
#             break
#         start = end - overlap
#         if start < 0:
#             start = 0
#     return out

# # -----------------------------
# def short_summary(text, max_chars=230):
#     if not text:
#         return ""
#     # prefer whole-sentence endings
#     text = text.replace("\n", " ").strip()
#     if len(text) <= max_chars:
#         return text
#     # split by punctuation to try end-of-sentence cut
#     import re
#     sentences = re.split(r'(?<=[\.\?\!])\s+', text)
#     out = ""
#     for s in sentences:
#         if not s:
#             continue
#         if len(out) + len(s) + 1 <= max_chars:
#             out = (out + " " + s).strip()
#         else:
#             break
#     if out:
#         return out if len(out) <= max_chars else out[:max_chars].rstrip() + "…"
#     # fallback: hard truncate
#     return text[:max_chars].rstrip() + "…"


# # -----------------------------
# def build_index():
#     global documents, chunks, vectorizer, chunk_vectors
#     documents = []
#     chunks = []
#     next_doc_id = 0

#     for ext in ALLOWED_EXTS:
#         for path in glob.glob(os.path.join(DOCS_DIR, ext)):
#             try:
#                 txt = load_file(path).strip()
#                 if not txt:
#                     print(f"no text extracted: {path}")
#                     continue
#                 documents.append({
#                     "id": next_doc_id,
#                     "path": path,
#                     "text": txt
#                 })
#                 doc_chunks = chunk_text(txt)
#                 for i, c in enumerate(doc_chunks):
#                     chunks.append({
#                         "doc_id": next_doc_id,
#                         "chunk_id": f"{next_doc_id}-{i}",
#                         "text": c
#                     })
#                 next_doc_id += 1
#             except Exception as e:
#                 print("error reading file", path, e)

#     if not chunks:
#         print("WARNING: No documents indexed. Add files to ./docs/")
#         vectorizer = None
#         chunk_vectors = None
#         return

#     chunk_texts = [c["text"] for c in chunks]
#     vectorizer = TfidfVectorizer(stop_words="english")
#     chunk_vectors = vectorizer.fit_transform(chunk_texts)
#     print(f"Indexed {len(documents)} documents -> {len(chunks)} chunks")

# # build at startup
# build_index()

# # -----------------------------
# # Flask app
# # -----------------------------
# app = Flask(__name__)
# CORS(app)

# @app.route("/", methods=["GET"])
# def health():
#     return "RAG short-answer backend running", 200

# @app.route("/reload", methods=["POST", "GET"])
# def reload_route():
#     build_index()
#     return jsonify({"status":"ok","docs":len(documents),"chunks":len(chunks)})

# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.get_json(force=True, silent=True) or {}
#     q = (data.get("question") or "").strip()
#     top_k = int(data.get("top_k") or 1)

#     if not q:
#         return jsonify({"answer": "Ask a question."})

#     if not chunks or vectorizer is None or chunk_vectors is None:
#         return jsonify({"answer": "No indexed documents found. Add files to ./docs and POST /reload."})

#     try:
#         qv = vectorizer.transform([q])
#         sims = cosine_similarity(qv, chunk_vectors)[0]
#         ranked = sims.argsort()[::-1]
#     except Exception as e:
#         return jsonify({"answer": f"Retrieval error: {e}"})

#     # get best chunk
#     best_excerpt = None
#     for idx in ranked:
#         if sims[idx] <= 0:
#             continue
#         best_excerpt = chunks[idx]["text"]
#         break

#     if not best_excerpt:
#         return jsonify({"answer": "I couldn't find anything relevant in your docs."})

#     # produce short answer
#     short = short_summary(best_excerpt, max_chars=230)
#     return jsonify({"answer": short})

# # if __name__ == "__main__":
# #     app.run(host="0.0.0.0", port=5000, debug=True)
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)