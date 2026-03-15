import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Force offline mode for National Sovereignty (Criterion A2)
os.environ["HF_HUB_OFFLINE"] = "1"

# --- CONFIGURATION ---
DATA_PATH = "data"
DB_PATH = "vectorstore/db_faiss"

print("📂 Loading documents from data folder...")

# Load PDFs
docs = []
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
    print(f"⚠️ Created missing {DATA_PATH} folder. Please add your PDFs there.")

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        print(f"  Loading: {file}")
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        docs.extend(loader.load())

if not docs:
    print("❌ No PDF documents found. Please add files to the 'data' folder.")
    exit()

print(f"✅ Loaded {len(docs)} document pages")

# Split documents
print("✂️ Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=300,  # 📈 Optimized for better retrieval quality
    length_function=len 
)

chunks = text_splitter.split_documents(docs) 
print(f"✅ Created {len(chunks)} text chunks")

# Create embeddings with offline mode
print("🧠 Creating embeddings (Local e5-base-v2)...")
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2",
    model_kwargs={'device': 'cpu', 'local_files_only': True},
    encode_kwargs={'normalize_embeddings': True}
)

# Create vectorstore
print("💾 Creating FAISS vectorstore...")
db = FAISS.from_documents(chunks, embeddings)

# Save to correct location
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
# Fix: Removed the allow_dangerous_deserialization argument here
db.save_local(DB_PATH) 

print(f"✅ Vectorstore saved to {DB_PATH}")
print("✅ Documents indexed successfully for Milestone 1 Review")
