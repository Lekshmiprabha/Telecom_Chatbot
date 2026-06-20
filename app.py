
"""
app.py
--------
Streamlit chatbot UI for the telecom RAG system.
Run: streamlit run app.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
try:
    import streamlit as st  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "The streamlit package is required. Install it with 'pip install streamlit'."
    ) from exc
from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR  = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_PER_SOURCE = 3
TOP_K_FINAL      = 5
CLAUDE_MODEL = "claude-sonnet-4-6"


@st.cache_resource
def load_vectorstores():
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


@st.cache_resource
def load_claude_client():
    return Anthropic()


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


def build_prompt(query: str, results: list, chat_history: list) -> str:
    context_blocks = []
    for doc, score, source_name in results:
        context_blocks.append(
            f"[Source: {source_name}]\n{doc.page_content}"
        )
    context_text = "\n\n---\n\n".join(context_blocks)

    history_text = ""
    if chat_history:
        history_lines = []
        for msg in chat_history:
            role = "Customer" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {msg['content']}")
        history_text = "\n".join(history_lines)

    prompt = f"""You are a helpful telecom customer support assistant.
Answer the customer's question using ONLY the information in the context below.
If the context does not contain enough information to answer, say so honestly
instead of guessing.

Conversation so far:
{history_text}

Context:
{context_text}

Customer question: {query}

Answer:"""

    return prompt


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
    st.title("📡 Telecom Support Chatbot")
    st.caption("Ask me about your data plan, billing, roaming, SIM issues, and more.")

    vectorstores = load_vectorstores()
    client = load_claude_client()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your question...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            results = search_all(vectorstores, user_input)
            prompt = build_prompt(user_input, results, st.session_state.messages[:-1])
            answer = generate_answer(client, prompt)

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()