import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings # <--- Updated

docs = []
for file in os.listdir("data"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join("data", file))
        docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base-v2"
)

db = FAISS.from_documents(chunks, embeddings)
db.save_local("vectorstore")

print("✅ Documents indexed successfully")
