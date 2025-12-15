from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
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


def run_interactive_search():
    print("\n" + "="*60)
    print("Simple Qdrant Semantic Search Demo")
    print("="*60)
    print("Type your query below. Leave blank to exit.")
    print("Add '|category:database' or '|category:AI' at the end to filter results.\n")

    while True:
        user_input = input("Your query: ").strip()
        if not user_input:
            print("Goodbye!")
            break

        query_text = user_input
        category_filter = None
        if "|" in user_input:
            parts = user_input.split("|", 1)
            query_text = parts[0].strip()
            filter_part = parts[1].strip().lower()
            if filter_part.startswith("category:"):
                category_filter = filter_part.split(":", 1)[1].strip()

        if not query_text:
            print("Please enter a query.\n")
            continue

        query_vector = model.encode(query_text).tolist()

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3,
            with_payload=True
        )

        print(f"\nSearch Results for: '{query_text}'")
        if category_filter:
            print(f"(Filtered by category: {category_filter})\n")
        else:
            print()

        match_count = 0
        for r in results.points:
            if category_filter and r.payload.get("category") != category_filter:
                continue
            match_count += 1
            print(f"- {r.payload['title']}")
            print(f"  Text: {r.payload['text']}")
            print(f"  Category: {r.payload['category']}")
            print(f"  Score: {r.score:.4f}\n")

        if match_count == 0:
            print("No matching results found.\n")

if __name__ == "__main__":
    ingest_documents()
    run_interactive_search()