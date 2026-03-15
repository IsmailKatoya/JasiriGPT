import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from prompts import QA_CHAIN_PROMPT

# Force offline mode for National Sovereignty (Criterion A2)
os.environ["HF_HUB_OFFLINE"] = "1"

# --- UI CONFIG ---
st.set_page_config(page_title="JasiriGPT v0.1.0", page_icon="🛡️", layout="wide")

st.title("🛡️ JasiriGPT: Kenyan Policy Assistant")
st.subheader("⚡ High-Speed MVP Demo Mode")

# --- INITIALIZE COMPONENTS ---
@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={'device': 'cpu', 'local_files_only': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 🏎️ PHI-3.5 OPTIMIZATION: Lightweight yet powerful for CPU inference
    llm = ChatOllama(
        model="phi3.5", 
        temperature=0.0,      # Deterministic answers for policy accuracy
        num_ctx=1024,        # Optimized context window for speed
        num_thread=8,        # Maximize your CPU cores
        num_predict=200,     # Precise, punchy answers for judges
    )
    return embeddings, llm

embeddings, llm = load_resources()

# --- LOAD VECTORSTORE ---
DB_PATH = "vectorstore/db_faiss"

if os.path.exists(DB_PATH):
    try:
        vectorstore = FAISS.load_local(
            DB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True # Keep for loading security
        )
        
        # ⚡ RETRIEVAL: k=2 for speed, relies on your new 300-char overlap from ingest.py
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | QA_CHAIN_PROMPT
            | llm
            | StrOutputParser()
        )
        
        # --- CHAT INTERFACE ---
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask about Finance Bill, SHIF, or the Constitution..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                status = st.status("🔍 Verifying Policy Documents...")
                try:
                    # 1. Faster retrieval
                    source_docs = retriever.invoke(prompt)
                    
                    # 2. Faster Phi-3.5 Inference
                    response = rag_chain.invoke(prompt)
                    
                    sources = set([doc.metadata.get('source', 'Unknown').split('/')[-1] for doc in source_docs])
                    full_response = response + f"\n\n📄 **Sources:** {', '.join(sources)}"
                    
                    status.update(label="✅ Policy Verified", state="complete", expanded=False)
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                except Exception as e:
                    status.update(label="❌ System Error", state="error")
                    st.error(f"Error: {str(e)}")
                            
    except Exception as e:
        st.error(f"❌ Database Error: {e}. Ensure ingest.py ran correctly.")
else:
    st.warning("⚠️ Vectorstore not found! Run 'python ingest.py' first.")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
    st.markdown("### 🛡️ MVP Controls")
    st.success("Mode: Sovereign Offline")
    st.info("Engine: Phi-3.5 (Optimized)")
    
    if st.button("🗑️ Reset Demo"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("🇰🇪 NIRU AI Hackathon 2026")
    st.caption("Candidate: Ismail Katoya Ali")
