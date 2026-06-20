# Telecom Chatbot RAG

A Retrieval-Augmented Generation (RAG) chatbot for telecom customer support using Chroma vector stores, HuggingFace embeddings, and Claude via Anthropic.

## Setup

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys. Example values are not tracked by git.

```env
ANTHROPIC_API_KEY=your_api_key_here
HF_TOKEN=your_hf_token_here
```

3. Make sure `.env` is excluded from version control via `.gitignore`.

## Ingestion

Run these scripts once to populate the Chroma collections:

```bash
python ingest_faq.py
python ingest_ticket.py
python Ingest_pdf.py
```

## Streamlit App

Start the UI with:

```bash
streamlit run app.py
```

## Notes

- The app and scripts use `langchain_chroma`, `langchain_core`, `langchain_huggingface`, and `anthropic`.
- The `chroma_store` directory is ignored in git and is used to persist vector stores locally.
- Do not commit `.env` or other sensitive credentials.
