
import os, json
from pathlib import Path
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from tqdm import tqdm


DOC_DIR = "docs"
COLLECTION_NAME = "my_docs"
EMB_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

def load_pdf(path):
    reader = PdfReader(path)
    parts = []
    for p in reader.pages:
        t = p.extract_text()
        if t:
            parts.append(t)
    return "\n\n".join(parts)

def load_docx(path):
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n\n".join(paragraphs)

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk_tokens))
        i += chunk_size - overlap
    return chunks

def main():
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_key = os.environ.get("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_key:
        raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY environment variables.")

    model = SentenceTransformer(EMB_MODEL)

    files = list(Path(DOC_DIR).glob("**/*.pdf")) + list(Path(DOC_DIR).glob("**/*.docx"))
    texts, metas = [], []
    for f in files:
        print("Loading", f)
        text = load_pdf(f) if f.suffix.lower() == ".pdf" else load_docx(f)
        if not text:
            continue
        chunks = split_text(text)
        for idx, c in enumerate(chunks):
            texts.append(c)
            metas.append({"source": str(f), "chunk": idx})

    if not texts:
        print("No documents found in docs/. Exiting.")
        return

    print("Encoding", len(texts), "chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

 
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, prefer_grpc=False)

    dim = embeddings.shape[1]
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        print("Collection exists:", COLLECTION_NAME)
    except Exception:
        print("Creating collection:", COLLECTION_NAME)
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

    batch = 64
    for i in tqdm(range(0, len(embeddings), batch)):
        j = min(i + batch, len(embeddings))
        ids = [f"doc_{k}" for k in range(i, j)]
        vecs = embeddings[i:j].tolist()
        payloads = [{"text": texts[k], **metas[k]} for k in range(i, j)]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                {"id": ids[idx], "vector": vecs[idx], "payload": payloads[idx]}
                for idx in range(len(ids))
            ]
        )

    with open("faiss_chunks.json", "w", encoding="utf-8") as fh:
        json.dump(texts, fh, ensure_ascii=False, indent=2)
    with open("faiss_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metas, fh, ensure_ascii=False, indent=2)

    print("Ingest complete. Pushed", len(texts), "chunks to Qdrant.")

if __name__ == "__main__":
    main()
