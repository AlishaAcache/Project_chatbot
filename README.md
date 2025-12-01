.. current update: proto4 ////

<img width="921" height="716" alt="image" src="https://github.com/user-attachments/assets/0b082e7b-ab1f-4ddf-aa48-a58548968217" />


<img width="351" height="550" alt="image" src="https://github.com/user-attachments/assets/654f8a42-72c5-4b6f-9322-246371ae8ab3" />


- Flask API with /chat, /reload, and /doc_summary.

- Loads documents from docs/ (.txt, .md, .pdf, .docx).

- Splits documents into chunks and builds a semantic index using sentence-transformers.

- Uses fast vector search (dot-product) to retrieve top-K relevant chunks.

- Includes an intent classifier + rule-based greeting handling (hi/hello/how are you).

- Synthesizes short answers from retrieved text (sentence ranking).

- Optionally improves the answer using OpenAI (enhance_with_llm()).

- Returns both the answer and source file paths.

- Configurable via env vars: OPENAI_API_KEY, OPENAI_MODEL, EMBEDDING_MODEL, TOP_K_DEFAULT, INTENT_CONF_THRESHOLD.

R — Retrieval

Finds relevant document text.
Where: build_index(), search_top_k()

A — Augmentation

Extracts the best sentences from retrieved chunks to form a concise context.
Where: synthesize_short_answer()

G — Generation

Produces the final natural answer (optionally using OpenAI).
Where: enhance_with_llm() + logic inside /chat

