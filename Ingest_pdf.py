import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pypdf import PdfReader

CHROMA_DIR    = "chroma_store"
COLLECTION    = "pdf"
PDF_PATH      = os.path.join("data", "telecom_guide.pdf")
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100
def load_pdf_documents(pdf_path: str) -> list[Document]:
    """Extract text page-by-page, then split into one Document per chunk."""
    reader = PdfReader(pdf_path)

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    docs = []
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if not page_text.strip():
            continue

        chunks = splitter.split_text(page_text)
        for chunk_idx, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": "pdf",
                    "page": page_num,
                    "chunk_index": chunk_idx,
                },
            ))
    return docs
def main():
    print("Loading and chunking PDF...")
    docs = load_pdf_documents(PDF_PATH)
    print(f"  {len(docs)} chunks created (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    print("Initialising embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print(f"Embedding and storing in Chroma collection '{COLLECTION}'...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Done. {vectorstore._collection.count()} vectors stored.")


if __name__ == "__main__":
    main()