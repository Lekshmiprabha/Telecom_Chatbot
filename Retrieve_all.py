import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR  = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_PER_SOURCE = 3
TOP_K_FINAL      = 5
def load_vectorstores() -> dict:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstores = {
        "faq": Chroma(
            collection_name="faq",
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        ),
        "tickets": Chroma(
            collection_name="tickets",
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        ),
        "pdf": Chroma(
            collection_name="pdf",
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        ),
    }
    return vectorstores
def search_all(vectorstores: dict, query: str,
               k_per_source: int = TOP_K_PER_SOURCE,
               k_final: int = TOP_K_FINAL):

    all_results = []

    for source_name, vectorstore in vectorstores.items():
        results = vectorstore.similarity_search_with_score(query, k=k_per_source)
        for doc, score in results:
            all_results.append((doc, score, source_name))

    # Sort everyone together by distance (lower = better match)
    all_results.sort(key=lambda item: item[1])

    return all_results[:k_final]
def main():
    print("Loading all vectorstores...")
    vectorstores = load_vectorstores()

    query = input("Ask a question: ")

    results = search_all(vectorstores, query)

    print(f"\nTop {len(results)} combined matches for: \"{query}\"\n")
    for i, (doc, score, source_name) in enumerate(results, start=1):
        print(f"--- Result {i} | source: {source_name} | distance: {score:.4f} ---")
        print(doc.page_content)
        print(f"Metadata: {doc.metadata}")
        print()


if __name__ == "__main__":
    main()