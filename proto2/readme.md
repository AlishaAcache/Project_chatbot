<img width="1278" height="630" alt="image" src="https://github.com/user-attachments/assets/4701b0d0-1d38-4aca-b30e-268e9fecb0aa" />
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
     ┌─────────────┴───────────────┐
     │ intent detected? (conf > 0.6)│
     └─────────────┬───────────────┘
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
