High-level map
Frontend: web-widget/index.html – the embeddable chat widget UI that talks to the orchestrator backend.
Backend services (under services/):
tenant-registry/ – who the tenant is (branding, limits, KB index, AI settings).
orchestrator-api/ – brain of the chatbot; routes each message to the right tool.
rag-service/ – knowledge base / RAG API for policy & document questions.
1. web-widget/index.html (frontend)
Role: Browser-side chat window that:
Loads tenant-specific config from the backend.
Sends user messages to the orchestrator.
Renders responses, suggested buttons, etc.
Main file to care about:
web-widget/index.html – contains HTML + JS for:
Initializing widget with tenantId
Calling /api/widget/config
Calling /api/chat/message for chat turns.
2. services/orchestrator-api (conversation brain)
Role: Entry point for all chat messages from any channel (widget, WhatsApp, email, etc.).
Validates tenantId via tenant middleware.
Detects intent (Azure OpenAI or regex).
Routes to tools: lead collection, property search, scheduling, knowledge base (RAG).
Optionally uses Azure OpenAI to polish the final reply.
Main files:
server.js – HTTP server & routes:
Defines /api/chat/message, /api/chat/history/:sessionId, /api/widget/config, /health.
Wires in middleware and calls processMessage from orchestrator.js.
middleware/tenant.js – tenant lookup for each request:
Reads x-tenant-id or query param.
Calls the tenant registry client to fetch tenant config and attaches it to the request/session.
services/orchestrator.js – core chatbot workflow:
Checks if lead collection is active → continues that flow.
Else, calls detectIntentWithAI (Azure OpenAI) → fallback detectIntent (regex).
Switches on intent: handleGreeting, handlePropertySearch, handleScheduling, handleKnowledgeQuery, handleGeneral.
For knowledge, calls RAG via rag-client.js and adds citations; then may call generateResponseWithAI for wording.
services/ai-intent.js – Azure OpenAI integration:
detectIntentWithAI – uses function calling to turn free text into structured intents + parameters.
generateResponseWithAI – optional LLM pass to rewrite/tool-based answers.
services/session.js – session management (in-memory or Cosmos in prod).
services/tenant-registry-client.js – HTTP client to the tenant-registry service.
services/rag-client.js – HTTP client to the RAG service (/api/knowledge/search).
lead-collection.js, lead-service.js, cosmos-lead-client.js – stateful lead form flow and persistence.
3. services/tenant-registry (tenant configuration + branding)
Role: Central config service for all tenants. Everything about “who this tenant is” lives here:
Branding (colors, logo, custom CSS).
Widget text & behavior (welcome message, position, enabled).
Channel enablement (WhatsApp, voice, etc.).
Knowledge base config (which index name).
AI config (model, deployment names, limits, etc.).
Main files:
server.js – HTTP server & routing:
Exposes /api/tenants, /api/tenants/:tenantId, /api/widget/config?tenantId=..., etc.
Connects to Cosmos DB (or in-memory in dev).
schema/tenant-schema.js – defines full tenant document structure:
tenantId, branding, widget, channels, knowledgeBase, ai, limits, security.
scripts/seed-tenants.js (in scripts/) – helper to create demo tenants in Cosmos.
4. services/rag-service (knowledge base / RAG API)
Role: Dedicated knowledge base microservice:
Upload & process docs (PDF, DOCX, TXT, HTML, Markdown).
Chunk text, call Azure OpenAI embeddings, and index into Azure Cognitive Search per tenant.
Handle search queries and optionally generate RAG answers with citations.
Main files:
server.js – HTTP server & routes:
/health, /api/knowledge/index/:tenantId, /api/knowledge/upload, /api/knowledge/search, /api/knowledge/status/:tenantId.
Wires in the RAG core logic from rag-service.js.
services/rag-service.js – main RAG pipeline:
Text extraction, chunking, embedding (Azure OpenAI), indexing/searching in Azure Cognitive Search.
Builds the final answer, sources, and confidence object returned to orchestrator.
services/rag-client.js – internal helper for talking to Azure Cognitive Search and Azure OpenAI (if you dig into the exact calls).
If you just want “the main entry points” to read/edit
Frontend: web-widget/index.html
Primary backend entry:
orchestrator-api/server.js
orchestrator-api/services/orchestrator.js
Tenant config:
tenant-registry/server.js
tenant-registry/schema/tenant-schema.js
Knowledge base / RAG:
rag-service/server.js
rag-service/services/rag-service.js
If you tell me what you want to customize (e.g., greeting behavior, lead flow, KB answers, widget look), I can point to the exact function(s) inside these files.

-----------------------------------


Tradeoff Check (Azure Cognitive Search vs. Azure + Qdrant)
Output accuracy
Qdrant combo wins: native vector index (HNSW, payload filters) lets you tune similarity, mix embeddings, and update in real time; Azure Cognitive Search’s semantic tier is solid but less configurable and slower to ingest freshly chunked chat or lead data.
Keep ACS only if keyword/legal compliance search is critical; otherwise Qdrant delivers better recall/precision on conversational answers.
Development time
ACS-only is faster: built-in vector fields, no extra cluster to wire up.
Adding Qdrant means provisioning AKS/container app, auth plumbing, new SDK layer, and CI/CD, so plan for extra weeks of infra + integration work.
Scalability
Qdrant on AKS scales elastically by shard/replica, supports multi-tenant isolation via payload filters or per-collection design, and handles high write rates better.
ACS scales too but each tier upgrade is coarse-grained and per-tenant isolation usually requires extra indexes; less flexible for hot ingestion workloads.
Development cost
ACS: just your existing services plus search index definitions—minimal additional spend.
Qdrant: engineering time for deployment manifests, health monitoring, backups, SDK integration; expect higher one-time cost.
Running cost
ACS pricing is predictable per tier, and you’re already paying for it.
Qdrant entails compute (AKS nodes or managed Qdrant), storage, backups, plus ops time; often higher monthly spend unless you shut down ACS’s vector features entirely.
Hybrid (ACS + Qdrant) is most expensive; consider replacing ACS vector search with Qdrant if budgets are tight.
Overall:
Need best possible RAG accuracy, fast freshness, fine-grained vector control → Azure + Qdrant is better despite higher dev/ops cost.
Need fastest delivery and lowest ongoing spend, acceptable accuracy using standard semantic search → stay with Azure Cognitive Search alone.

# Vector Database Comparison Table

| Vector DB                                     | Open-Source?       | Deployment                            | Key Merits                                                                                                            | Drawbacks                                                          |
| --------------------------------------------- | ------------------ | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Qdrant**                                    | ✔ Yes (Apache-2.0) | Self-host / Cloud                     | Very fast ANN search (HNSW), excellent metadata filtering, hybrid search, easy API, cost-efficient, production-ready. | Not fully distributed like Milvus for *extreme* scale (billions).  |
| **Weaviate**                                  | ✔ Yes              | Self-host / Cloud                     | Schema-based, hybrid search, good for structured + unstructured mix, GraphQL API.                                     | Heavier setup; not as lightweight as Qdrant.                       |
| **Milvus**                                    | ✔ Yes              | Self-host / Kubernetes / Zilliz Cloud | Massive scale (billions+), distributed, high-performance ANN.                                                         | Overkill for mid-size projects; complex infra.                     |
| **Zilliz Cloud** (Milvus Cloud)               | ✖ No (managed)     | Managed Cloud                         | Milvus power without infra management.                                                                                | Vendor pricing; less control.                                      |
| **Pinecone**                                  | ✖ No               | Managed Cloud                         | Easiest to use; reliable, fast; strong production hosting.                                                            | Closed-source; expensive at scale; cannot self-host.               |
| **ElasticSearch + Vector**                    | ✔ Yes              | Self-host / Cloud                     | Good if you already use Elastic; combines keyword + vector search.                                                    | Not a purpose-built vector DB; slower for large vector workloads.  |
| **Azure Cosmos DB for MongoDB + Vector**      | ✖ No               | Managed Cloud                         | Good when you already use Cosmos DB; integrates vector search into existing workloads.                                | Not optimized for heavy vector workloads.                          |
| **Azure DocumentDB w/ MongoDB Compatibility** | ✖ No               | Managed Cloud                         | Easy if already using Azure Mongo API.                                                                                | Basic vector support; not ideal for large-scale similarity search. |
| **DataStax Astra / HCD**                      | ✔ Core open-source | Self-host / Cloud                     | Cassandra + vector search; scalable, good for enterprise workloads.                                                   | Not as specialized for ANN as Qdrant/Milvus.                       |
| **Tiger Data / TigerGraph**                   | ✖ No               | Managed                               | Graph DB + vector. Good for graph + embeddings combo.                                                                 | Heavy, complex; niche use cases.                                   |
| **SingleStore Helios**                        | ✖ No               | Managed Cloud                         | Hybrid SQL + vector search, fast OLAP + vector mix.                                                                   | Not a pure vector DB; cost.                                        |
| **Kimera**                                    | ✖ No               | SaaS                                  | More of a tagging/AI content platform than vector DB.                                                                 | Not comparable to dedicated vector DBs.                            |
