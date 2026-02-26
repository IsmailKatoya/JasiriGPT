import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from prompts import QA_CHAIN_PROMPT

# Force offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

# --- UI CONFIG ---
st.set_page_config(page_title="JasiriGPT v0.1.0", page_icon="🛡️", layout="wide")

st.title("🛡️ JasiriGPT: Kenyan Policy Assistant")
st.subheader("v0.1.0 - Milestone 1 Release (Quality Focus)")

# --- INITIALIZE COMPONENTS ---
@st.cache_resource
def load_resources():
    # High-quality embeddings (768-dim)
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={'device': 'cpu', 'local_files_only': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Sustainable Mistral Config: Capped threads to protect hardware
    llm = ChatOllama(
        model="mistral",
        temperature=0.1,    # Slight creativity for better language flow
        num_ctx=2048,       # Full context for better policy reading
        num_thread=2,       # 🛡️ Thermal safety: cap CPU usage
        num_predict=256,    # Allow for complete, high-quality answers
        repeat_penalty=1.2
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
        
        # QUALITY SEARCH: Retrieve top 3 chunks for better context
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # RAG Chain
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
                status = st.status("🔍 Deep Policy Analysis Active...")
                try:
                    status.write("📖 Reading legal documents...")
                    source_docs = retriever.invoke(prompt)
                    
                    status.write("🤖 Thinking (Thermal Safety Mode)...")
                    response = rag_chain.invoke(prompt)
                    
                    sources = set([
                        doc.metadata.get('source', 'Unknown').split('/')[-1] 
                        for doc in source_docs
                    ])
                    source_text = f"\n\n📄 **Sources:** {', '.join(sources)}"
                    
                    full_response = response + source_text
                    status.update(label="✅ Analysis Complete", state="complete", expanded=False)
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(f"Error during inference: {str(e)}")
                            
    except Exception as e:
        st.error(f"❌ Database Error: {e}. Try running ingest.py again.")
else:
    st.warning("⚠️ Vectorstore not found! Run: python ingest.py")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.markdown("### 🛡️ System Status")
    st.success("✅ Mistral: Quality Mode")
    st.info("CPU: Sustainability Cap (2-Thread)")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("🇰🇪 NIRU 2026 | Ismail Katoya Ali")
    st.caption("v0.1.0 Milestone 1 Release")
