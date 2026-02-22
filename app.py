import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama  # Switched for faster streaming

# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

# --- UI CONFIG ---
st.set_page_config(page_title="JasiriGPT v0.1.0", page_icon="🛡️", layout="centered")
st.title("🛡️ JasiriGPT: Milestone 1")
st.caption("Sovereign AI | High-Speed Optimized | v0.1.0 Stable")

# --- 1. CACHED RESOURCE LOADING (Lead Dev Fix) ---
@st.cache_resource
def load_resources():
    # Lightest embeddings for CPU
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu', 'local_files_only': True}
    )
    # Direct Ollama wrapper is faster than ChatOllama for streaming
    llm = Ollama(
        model="phi3:mini", 
        temperature=0.1,
        num_ctx=2048,
        num_predict=256,
        num_thread=4 
    )
    return embeddings, llm

@st.cache_resource
def load_vectorstore(_embeddings):
    DB_PATH = "vectorstore/db_faiss"
    if os.path.exists(DB_PATH):
        return FAISS.load_local(DB_PATH, _embeddings, allow_dangerous_deserialization=True)
    return None

# Initialize
embeddings, llm = load_resources()
vectorstore = load_vectorstore(embeddings)

# --- 2. THE INFERENCE ENGINE (Streaming Fix) ---
def answer_question(question):
    # Retrieve top 2 chunks (Speed/Context balance)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(question)
    
    # Build context string
    context = "\n\n".join([d.page_content[:1000] for d in docs])
    
    # Simplified Prompt for v0.1.0 Speed
    prompt = f"""You are JasiriGPT, a Kenyan Policy Expert. 
Use the context to answer in English. 
Context: {context}
Question: {question}
Answer:"""
    
    return llm.stream(prompt), docs

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Ask about SHIF or Finance Act..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    if vectorstore:
        with st.chat_message("assistant"):
            # This 'status' bar gives the judges a visual cue that it's working
            status = st.status("⚡ Sovereign AI is searching documents...")
            
            # Start streaming the response
            full_response = ""
            placeholder = st.empty()
            
            stream, docs = answer_question(prompt)
            
            status.write("📖 Context found. Generating answer...")
            for chunk in stream:
                full_response += chunk
                placeholder.markdown(full_response + "▌") # Animated cursor
            
            placeholder.markdown(full_response) # Final static text
            
            # Show sources
            sources = {d.metadata.get('source', 'Policy').split('/')[-1] for d in docs}
            st.markdown(f"\n\n**Source Documents:** `{', '.join(sources)}`")
            
            status.update(label="✅ Analysis Complete", state="complete", expanded=False)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.error("System Error: Vectorstore not found. Please run ingest.py.")
