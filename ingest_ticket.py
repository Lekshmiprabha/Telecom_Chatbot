import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import sqlite3
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR  = "chroma_store"
COLLECTION  = "tickets"
DB_PATH     = os.path.join("data", "tickets.db")
TABLE_NAME  = "tickets"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
def load_ticket_documents(db_path: str, table_name: str) -> list[Document]:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()

    docs = []
    for _, row in df.iterrows():
        content = (
            f"Issue Type: {row['issue_type']}\n"
            f"Description: {row['description']}\n"
            f"Resolution: {row['resolution']}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "source": "ticket",
                "ticket_id": str(row["ticket_id"]),
                "category": row["category"],
                "status": row["status"],
            },
        ))
    return docs
def main():
    print("Loading ticket documents...")
    docs = load_ticket_documents(DB_PATH, TABLE_NAME)
    print(f"  {len(docs)} ticket entries loaded.")

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