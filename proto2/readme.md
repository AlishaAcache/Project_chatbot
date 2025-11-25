STEPS TO RUN : 
- 
---- we are using only ... app.py, index.html, docs/,requirements.txt
- pip install -r requirements.txt
- pip install "sentence-transformers[torch]" faiss-cpu
- python app.py
  After all the above steps go to index.html ---> run on live server.
 
----------------------------------------------------
<img width="1278" height="630" alt="image" src="https://github.com/user-attachments/assets/4701b0d0-1d38-4aca-b30e-268e9fecb0aa" />

# RAG Pipeline used here----
R → Retrieval (finding relevant text from your documents)

- results = search_top_k(q, k=top_k)
  
A → Augmented (using that retrieved text as context)

- combined_text = " ".join([r["text"] for r in results])

G → Generation (creating a short natural-language answer from it)
- synthesize_short_answer()

  

                ┌─────────────────────────────────────────────┐
                │               STARTUP (app run)              │
                └─────────────────────────────────────────────┘
                                  │
                                  ▼
                   ┌────────────────────────────┐
                   │ build_index()              │
                   └────────────────────────────┘
                        │   Loads all docs
                        │   Reads .pdf/.docx/.txt
                        │   Splits into chunks
                        │   Creates embeddings
                        ▼
          ┌──────────────────────────────┐
          │ documents[]                  │
          │ chunks[]                     │
          │ chunk_embeddings (vectors)   │
          └──────────────────────────────┘
                        │
                        ▼
           ┌────────────────────────────────┐
           │ train_intent_classifier()      │
           └────────────────────────────────┘


                        USER ASKS A QUESTION

                ┌─────────────────────────────────┐
                │ /chat endpoint receives request │
                └─────────────────────────────────┘
                                  │
                                  ▼
                ┌────────────────────────────────┐
                │ predict_intent(question)       │
                └────────────────────────────────┘
                   │
                ┌────────────────────────────────┐
                │ intent detected?(conf > 0.6)    │
                └────────────────────────────────┘
     
                   │yes                         │no → go to RAG
                   ▼
       returns canned intent reply





                      RAG PIPELINE STARTS

                     ┌────────────────────┐
        query ─────▶ │ search_top_k()     │
                     └────────────────────┘
                      │   embeds query
                      │   cosine similarity
                      │   returns top-k chunks
                      ▼
          ┌──────────────────────────────┐
          │ retrieved_chunks (text + path) │
          └──────────────────────────────┘
                      │
                      ▼
            ┌──────────────────────────────┐
            │ synthesize_short_answer()     │
            └──────────────────────────────┘
               • merges retrieved chunk text  
               • splits into sentences  
               • scores sentences with embeddings  
               • picks top 1–2  
               • forms conversational answer  
                      │
                      ▼
        ┌─────────────────────────────────────────┐
        │ final answer + sources (paths of chunks)│
        └─────────────────────────────────────────┘
                      │
                      ▼
               returned to frontend
