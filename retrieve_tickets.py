"""
retrieve_tickets.py
----------------------
Interactively searches the 'tickets' Chroma collection for the most
relevant past tickets given a user's typed question.
Run: python retrieve_tickets.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR  = "chroma_store"
COLLECTION  = "tickets"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K       = 3


def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


def search_tickets(vectorstore: Chroma, query: str, k: int = TOP_K):
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


def main():
    print("Loading ticket vectorstore...")
    vectorstore = load_vectorstore()

    query = input("Ask a question: ")

    results = search_tickets(vectorstore, query)

    print(f"\nTop {len(results)} matches for: \"{query}\"\n")
    for i, (doc, score) in enumerate(results, start=1):
        print(f"--- Result {i} (distance: {score:.4f}) ---")
        print(doc.page_content)
        print(f"Metadata: {doc.metadata}")
        print()


if __name__ == "__main__":
    main()