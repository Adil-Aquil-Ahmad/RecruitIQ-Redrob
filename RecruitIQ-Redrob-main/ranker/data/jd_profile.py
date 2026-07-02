"""
Structured representation of the Senior AI Engineer JD.
All vocabulary lists are derived from the actual job description text.
"""

JD_TEXT_KEY_SECTIONS = """
Senior AI Engineer Founding Team Redrob AI Series A production embeddings
retrieval semantic search vector database ranking recommendation system
python machine learning deep learning nlp transformer bert fine-tuning lora peft rag
faiss pinecone qdrant weaviate elasticsearch opensearch milvus
learning to rank ndcg mrr map a/b testing evaluation framework
deployed production inference pipeline scale latency real users
sentence transformers dense retrieval hybrid search approximate nearest neighbor
pytorch tensorflow huggingface startup product company applied ml
"""

# Expanded BM25 query with synonyms for vocabulary mismatch coverage
BM25_QUERY_TOKENS = """
embedding embeddings vector vectors dense retrieval semantic search neural search
faiss pinecone qdrant weaviate elasticsearch opensearch milvus chroma pgvector
approximate nearest neighbor ann hnsw index
ranking ranker recommendation collaborative filtering learning to rank ltr l2r xgboost lightgbm
nlp natural language processing bert transformer huggingface sentence transformer
pytorch tensorflow jax deep learning neural network
fine tuning finetuning lora qlora peft adapter instruction tuning
rag retrieval augmented generation llm large language model gpt inference
ndcg mrr map precision recall evaluation ab testing abtesting online experiment
production deployed serving pipeline scale latency throughput real time
python scikit sklearn pandas numpy scipy
startup product company applied ml machine learning engineer ai engineer data scientist
search engineer recommendation system matching relevance
""".split()

# Core JD required skills — used for evidence scoring
REQUIRED_SKILL_VOCAB = [
    "embedding", "embeddings", "vector", "sentence-transformer",
    "faiss", "pinecone", "qdrant", "weaviate", "elasticsearch", "opensearch",
    "retrieval", "semantic search", "dense retrieval", "hybrid search",
    "ranking", "recommendation", "learning to rank",
    "nlp", "bert", "transformer", "huggingface",
    "pytorch", "tensorflow",
    "fine-tuning", "lora", "peft", "rag",
    "ndcg", "mrr", "evaluation", "a/b",
    "python",
    "machine learning", "deep learning",
]

# Skill names from the skills field that match JD requirements
JD_SKILL_NAMES = {
    "python", "machine learning", "deep learning", "nlp", "natural language processing",
    "bert", "transformers", "pytorch", "tensorflow", "scikit-learn", "sklearn",
    "embeddings", "vector search", "semantic search", "faiss", "elasticsearch",
    "qdrant", "pinecone", "weaviate", "milvus", "opensearch",
    "recommendation systems", "collaborative filtering",
    "ranking", "information retrieval",
    "fine-tuning", "llm", "large language model", "rag",
    "a/b testing", "experimentation",
    "mlops", "model deployment", "model serving",
    "huggingface", "sentence transformers",
    "spark", "kafka", "airflow",
    "sql", "data engineering",
}

# JD ideal candidate profile
IDEAL_PROFILE = {
    "yoe_optimal": 6.5,
    "yoe_range": (5, 9),
    "preferred_locations": {
        "noida", "pune", "hyderabad", "bangalore", "bengaluru",
        "mumbai", "delhi", "delhi ncr", "gurgaon", "gurugram",
    },
    "preferred_countries": {"india"},
    "preferred_work_modes": {"hybrid", "onsite", "flexible"},
}
