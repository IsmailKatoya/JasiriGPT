import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
DATA_PATH = "data"
DB_PATH = "vectorstore/db_faiss"  # Match app.py exactly

print("📂 Loading documents from data folder...")

# Load PDFs
docs = []
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        print(f"  Loading: {file}")
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        docs.extend(loader.load())

print(f"✅ Loaded {len(docs)} document pages")

# Split documents
print("✂️ Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = splitter.split_documents(docs)
print(f"✅ Created {len(chunks)} text chunks")

# Create embeddings
print("🧠 Creating embeddings (this may take a while)...")
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2"
)

# Create vectorstore
print("💾 Creating FAISS vectorstore...")
db = FAISS.from_documents(chunks, embeddings)

# Save to correct location
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
db.save_local(DB_PATH)

print(f"✅ Vectorstore saved to {DB_PATH}")
print("✅ Documents indexed successfully")
