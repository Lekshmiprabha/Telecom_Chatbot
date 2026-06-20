import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from dotenv import load_dotenv
load_dotenv()  # reads .env file and loads its variables into the environment

from anthropic import Anthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR  = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_PER_SOURCE = 3
TOP_K_FINAL      = 5

CLAUDE_MODEL = "claude-sonnet-4-6"
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - provide clear error if package missing
    class Anthropic:  # minimal stub to provide helpful import-time message
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The 'anthropic' package is not installed or could not be imported. "
                "Install it with: pip install anthropic"
            )
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR  = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_PER_SOURCE = 3
TOP_K_FINAL      = 5

CLAUDE_MODEL = "claude-sonnet-4-6"
def build_prompt(query: str, results: list) -> str:
    context_blocks = []
    for doc, score, source_name in results:
        context_blocks.append(
            f"[Source: {source_name}]\n{doc.page_content}"
        )

    context_text = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a helpful telecom customer support assistant.
Answer the customer's question using ONLY the information in the context below.
If the context does not contain enough information to answer, say so honestly
instead of guessing.

Context:
{context_text}

Customer question: {query}

Answer:"""

    return prompt
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

    all_results.sort(key=lambda item: item[1])

    return all_results[:k_final]
def generate_answer(client: Anthropic, prompt: str) -> str:
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=500,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    return response.content[0].text
def main():
    print("Loading vectorstores...")
    vectorstores = load_vectorstores()

    client = Anthropic()  # reads ANTHROPIC_API_KEY from environment automatically

    query = input("Ask a question: ")

    print("\nRetrieving relevant context...")
    results = search_all(vectorstores, query)

    print("Building prompt...")
    prompt = build_prompt(query, results)

    print("Asking Claude...\n")
    answer = generate_answer(client, prompt)

    print("=== Answer ===")
    print(answer)

    print("\n=== Sources used ===")
    for doc, score, source_name in results:
        print(f"- {source_name} (distance: {score:.4f})")


if __name__ == "__main__":
    main()