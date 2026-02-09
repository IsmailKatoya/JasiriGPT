import os
# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import our optimized prompt logic
from prompts import QA_CHAIN_PROMPT

# --- CONFIGURATION & UI ---
st.set_page_config(page_title="JasiriGPT Sovereign AI", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatFloatingInputContainer { bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ JasiriGPT: Kenyan Policy Assistant")
st.subheader("Sovereign AI Prototype (Optimized) - NIRU 2026")

# --- INITIALIZE COMPONENTS ---
@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={'device': 'cpu', 'local_files_only': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # PERFORMANCE TUNED: Lower context and prediction limit for faster local CPU inference
    llm = ChatOllama(
        model="mistral",
        temperature=0,
        num_predict=250,  # Slightly higher than 200 to ensure Kiswahili isn't cut off
        num_ctx=2048      
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
            allow_dangerous_deserialization=True
        )
        
        # SPEED OPTIMIZED: Retrieval limited to top 2 chunks
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 2} 
        )
        
        def format_docs(docs):
            # Limit each chunk to 1000 chars to stay within the faster context window
            return "\n\n---\n\n".join(doc.page_content[:1000] for doc in docs)
        
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
        
        if prompt := st.chat_input("Ask about Finance Act, SHIF, or Constitution..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                status = st.status("⚡ Fast Processing...")
                try:
                    status.write("📖 Quick Retrieval...")
                    source_docs = retriever.invoke(prompt)
                    
                    status.write("🤖 Generating...")
                    response = rag_chain.invoke(prompt)
                    
                    sources = set([
                        doc.metadata.get('source', 'Unknown').split('/')[-1] 
                        for doc in source_docs
                    ])
                    source_text = f"\n\n📄 **Sources:** {', '.join(sources)}"
                    
                    full_response = response + source_text
                    status.update(label="✅ Done", state="complete", expanded=False)
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response
                    })
                    
                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(f"Error: {str(e)}")
                        
    except Exception as e:
        st.error(f"❌ Error loading database: {e}")
else:
    st.warning("⚠️ Vectorstore not found! Run: python ingest.py")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.markdown("### 🛡️ System Status")
    st.success("✅ Offline & Speed Optimized")
    
    st.markdown("### 📚 Knowledge Base")
    st.info("- Finance Act 2024\n- SHIF Regs 2024\n- Constitution 2010")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("🇰🇪 NIRU 2026 | Ismail Katoya Ali")
