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
