import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder  

from config import (
    RETRIEVER_POOL_SIZE,
    TOP_K,
    DB_FOLDER
)

# ======================================================
# LOAD MODELS & DATABASE
# ======================================================
if not os.path.exists(DB_FOLDER):
    print(f"ERROR: Database folder '{DB_FOLDER}' not found.")
    print("Run 'create_vector_db.py' first!")
    exit()

print("Loading embedding model (bi-encoder)...")
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading Chroma database...")
db = Chroma(persist_directory=DB_FOLDER, embedding_function=embedding_function)

print("Loading Cross-Encoder reranker (BAAI/bge-reranker-base)...")
reranker = CrossEncoder("BAAI/bge-reranker-base")

# ======================================================
# RETRIEVAL FUNCTIONS
# ======================================================
def retrieve_candidates(query, pool_size):
    """Retrieve many candidate chunks using vector search."""
    return db.similarity_search(query, k=pool_size)


def rerank_with_cross_encoder(query, candidates, top_k):
    """Rerank candidate chunks using a cross-encoder model."""
    # Prepare (query, doc) pairs
    pairs = [(query, doc.page_content) for doc in candidates]

    # Get scores from the cross encoder
    scores = reranker.predict(pairs)

    # Combine documents and scores
    scored_docs = list(zip(candidates, scores))

    # Sort by score descending and select top_k
    reranked = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    top_docs = reranked[:top_k]

    # Return the documents only (in sorted order)
    return [doc for doc, score in top_docs]


# ======================================================
# INTERACTIVE SEARCH LOOP
# ======================================================
if __name__ == "__main__":
    # interactive loop here

    print("\n" + "="*50)
    print("        DRUG KNOWLEDGE BASE (2-Stage RAG)")
    print("="*50)

    while True:
        query = input("\nAsk a question (or 'q' to quit): ")
        if query.lower() == 'q':
            break

        # ------------------------------------------------------
        # STAGE 1 — Bi-Encoder Retrieval
        # ------------------------------------------------------
        print(f"\n[Stage 1] Retrieving {RETRIEVER_POOL_SIZE} candidates using vector similarity...")
        candidates = retrieve_candidates(query, RETRIEVER_POOL_SIZE)

        # Safety check
        if not candidates:
            print("No results found.")
            continue

        # ------------------------------------------------------
        # STAGE 2 — Cross-Encoder Reranking
        # ------------------------------------------------------
        print(f"[Stage 2] Reranking candidates with Cross-Encoder (selecting Top {TOP_K})...")
        top_chunks = rerank_with_cross_encoder(query, candidates, TOP_K)

        print(f"\n--- Final Top {len(top_chunks)} Relevant Chunks ---")

        # Print only the final reranked chunks
        for i, doc in enumerate(top_chunks):
            drug_name = doc.metadata.get('drug_name', 'Unknown')
            type_ = doc.metadata.get('type', 'General')

            print(f"\n[Result {i+1}] Drug: {drug_name} ({type_})")
            print("-" * 30)
            print(doc.page_content[:350] + "...")
            print("-" * 30)

    print("\nExiting search.")


__all__ = ["retrieve_candidates", "rerank_with_cross_encoder"]


