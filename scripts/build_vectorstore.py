import os
import json

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document  # use community schema
CHUNKS_PATH = "data/constitution_chunks.json"
VECTORSTORE_PATH = "vectorstore"

# Load chunks
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

docs = [Document(page_content=chunk) for chunk in chunks]

print("🧠 Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("📦 Building FAISS vectorstore...")
db = FAISS.from_documents(docs, embeddings)

db.save_local(VECTORSTORE_PATH)
print(f"✅ Vectorstore saved at {VECTORSTORE_PATH}")
