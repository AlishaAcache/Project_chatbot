
# USING OPEN AI 
# import os
# import glob
# import time
# from flask import Flask, request, jsonify
# from flask_cors import CORS

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# from PyPDF2 import PdfReader
# from docx import Document

# # Optional: openai client
# import openai

# # -----------------------------
# # Config (tweak these)
# # -----------------------------
# DOCS_DIR = "docs"
# ALLOWED_EXTS = ("*.txt", "*.md", "*.pdf", "*.docx")
# CHUNK_SIZE = 800
# CHUNK_OVERLAP = 200
# SIM_THRESHOLD = 0.03    # minimum similarity to consider calling LLM
# OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # set this in Render / env

# if OPENAI_API_KEY:
#     openai.api_key = OPENAI_API_KEY

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
# # Short safe fallback summary (no LLM)
# # -----------------------------
# def short_summary(text, max_chars=220):
#     if not text:
#         return ""
#     text = text.replace("\n", " ").strip()
#     if len(text) <= max_chars:
#         return text
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
#     return text[:max_chars].rstrip() + "…"

# # -----------------------------
# # Index builder
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

# # initial index
# build_index()

# # -----------------------------
# # OpenAI helper (RAG prompt)
# # -----------------------------
# def call_openai_summarize(question, retrieved_texts):
#     """
#     Call OpenAI ChatCompletion with a tight system prompt forcing
#     short, source-free answers. Returns the assistant's reply string.
#     """
#     if not OPENAI_API_KEY:
#         return None  # caller will fallback
#     # Safety: limit context length (join only top few chunks)
#     context = "\n\n".join(retrieved_texts[:4])
#     system = (
#         "You are a concise assistant that answers only from the provided CONTEXT. "
#         "Answer in 1-3 short sentences. Do NOT mention filenames, sources, or where the information came from. "
#         "If the answer cannot be found in the CONTEXT, reply exactly: 'I don't have that information in the documents.'"
#     )
#     user = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nProvide a brief answer (1-3 short sentences)."

#     try:
#         resp = openai.ChatCompletion.create(
#             model=OPENAI_MODEL,
#             messages=[
#                 {"role": "system", "content": system},
#                 {"role": "user", "content": user}
#             ],
#             max_tokens=150,
#             temperature=0.0,
#             top_p=1.0,
#             n=1
#         )
#         return resp["choices"][0]["message"]["content"].strip()
#     except Exception as e:
#         print("OpenAI call failed:", e)
#         return None

# # -----------------------------
# # Flask app
# # -----------------------------
# app = Flask(__name__)
# CORS(app)

# @app.route("/", methods=["GET"])
# def health():
#     return "RAG+LLM backend running", 200

# @app.route("/reload", methods=["POST", "GET"])
# def reload_route():
#     build_index()
#     return jsonify({"status":"ok","docs":len(documents),"chunks":len(chunks)})

# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.get_json(force=True, silent=True) or {}
#     question = (data.get("question") or "").strip()
#     top_k = int(data.get("top_k") or 3)

#     if not question:
#         return jsonify({"answer": "Ask a question."})

#     if not chunks or vectorizer is None or chunk_vectors is None:
#         return jsonify({"answer": "No indexed documents found. Add files to ./docs and POST /reload."})

#     # retrieval
#     try:
#         qv = vectorizer.transform([question])
#         sims = cosine_similarity(qv, chunk_vectors)[0]
#         ranked = sims.argsort()[::-1]
#     except Exception as e:
#         return jsonify({"answer": f"Retrieval error: {e}"})

#     # collect top_k retrieved chunks and their scores
#     retrieved_texts = []
#     top_scores = []
#     for idx in ranked[:max(10, top_k*2)]:  # check a few to allow threshold filtering
#         if sims[idx] <= 0:
#             continue
#         retrieved_texts.append(chunks[idx]["text"])
#         top_scores.append(float(sims[idx]))

#     if not retrieved_texts:
#         return jsonify({"answer": "I couldn't find anything relevant in your documents."})

#     # decide whether retrieval is strong enough
#     max_sim = max(top_scores) if top_scores else 0.0
#     if max_sim < SIM_THRESHOLD:
#         # don't call LLM for weak matches; safer to say unknown
#         return jsonify({"answer": "I don't have that information in the documents."})

#     # call LLM (if available) with top retrieved texts
#     answer = call_openai_summarize(question, retrieved_texts[:top_k])

#     # fallback: if OpenAI not configured or failed, return short summary of best chunk
#     if not answer:
#         answer = short_summary(retrieved_texts[0], max_chars=220)

#     # final: ensure short single-line answer (truncate if necessary)
#     answer = " ".join(answer.splitlines()).strip()
#     if len(answer) > 400:
#         answer = answer[:400].rstrip() + "…"

#     return jsonify({"answer": answer})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
