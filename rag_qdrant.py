from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue, QueryResponse
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "documents"

client = QdrantClient(url="http://localhost:6333")

model = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_DIM = model.get_sentence_embedding_dimension()

def ingest_documents():
    docs = [
        {"title": "Vector Databases", "text": "Vector databases store embeddings of text for semantic search.", "category": "database"},
        {"title": "Python RAG", "text": "RAG combines retrieval and generation using language models.", "category": "AI"},
        {"title": "Qdrant Overview", "text": "Qdrant is a vector database optimized for embeddings.", "category": "database"},
    ]

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        print(f"Collection created: {COLLECTION_NAME}")

    vectors = [model.encode(doc["text"]).tolist() for doc in docs]
    points = [
        {
            "id": idx,
            "vector": vectors[idx],
            "payload": doc
        } for idx, doc in enumerate(docs)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("Documents ingested with metadata")


def semantic_search(query: str, top_k: int = 3):
    query_vector = model.encode(query).tolist()

    results: QueryResponse = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,        
        limit=top_k,
        with_payload=True,
        with_vectors=False          
    )

    print(f"\nSearch Results for: '{query}'")
    for r in results.points:
        print(f"- {r.payload['title']} | score={r.score:.4f} | category={r.payload['category']}")


def filtered_search(query: str, category: str, top_k: int = 3):
    query_vector = model.encode(query).tolist()

    results: QueryResponse = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,        
        query_filter=Filter(       
            must=[
                FieldCondition(
                    key="category",
                    match=MatchValue(value=category)
                )
            ]
        ),
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )

    print(f"\nFiltered Search ({category}) for: '{query}'")
    for r in results.points:
        print(f"- {r.payload['title']} | score={r.score:.4f}")


if __name__ == "__main__":
    ingest_documents()
    semantic_search("How do vector databases work?")
    filtered_search("How do vector databases work?", category="database")