import os
# Force offline mode for HuggingFace before importing LangChain components
os.environ["HF_HUB_OFFLINE"] = "1"

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import the centralized prompt from our module
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
st.subheader("Sovereign AI Prototype - NIRU 2026")

# --- INITIALIZE COMPONENTS ---
@st.cache_resource
def load_resources():
    # Load embeddings with strict local paths
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={'device': 'cpu', 'local_files_only': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Initialize Local Ollama Instance
    llm = ChatOllama(
        model="mistral",
        temperature=0,
        num_predict=400,
        num_ctx=3072
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
        
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        def format_docs(docs):
            return "\n\n---\n\n".join(doc.page_content for doc in docs)
        
        # LCEL Chain - Modular and Scalable
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
                status = st.status("🔍 Processing...")
                try:
                    status.write("📖 Searching sovereign documents...")
                    source_docs = retriever.invoke(prompt)
                    
                    status.write("🤖 Generating expert response...")
                    response = rag_chain.invoke(prompt)
                    
                    # Extract source filenames for transparency
                    sources = set([
                        doc.metadata.get('source', 'Unknown').split('/')[-1] 
                        for doc in source_docs
                    ])
                    source_text = f"\n\n📄 **Sources:** {', '.join(sources)}"
                    
                    full_response = response + source_text
                    status.update(label="✅ Complete", state="complete", expanded=False)
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response
                    })
                    
                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(f"Error during inference: {str(e)}")
                        
    except Exception as e:
        st.error(f"❌ Error loading database: {e}")
else:
    st.warning("⚠️ Vectorstore not found! Run: python ingest.py")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.markdown("### 🛡️ System Status")
    st.success("✅ Offline Mode Active")
    
    st.markdown("### 📚 Documents Loaded")
    st.info("""
    - Finance Act 2024
    - SHIF Regulations 2024
    - Constitution of Kenya 2010
    """)
    
    st.markdown("### 💡 Tips")
    st.caption("Ask specific policy questions:")
    st.code("""
- How do I register for SHIF?
- Who is eligible for SHIF?
- What does Article 43 say?
    """, language="text")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("🇰🇪 NIRU 2026 AI Challenge")
    st.caption("Ismail Katoya Ali | HAI-2026-007")
