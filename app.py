import os
os.environ["HF_HUB_OFFLINE"] = "1"

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={'device': 'cpu', 'local_files_only': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    llm = ChatOllama(
        model="mistral",
        temperature=0,
        num_predict=400,
        num_ctx=3072
    )
    return embeddings, llm

embeddings, llm = load_resources()

# --- IMPROVED BILINGUAL PROMPT ---
template = """You are JasiriGPT, a Kenyan Government Policy Expert.

INSTRUCTIONS:
1. Provide a clear, accurate answer in English based on the context
2. Use numbered steps or bullet points for procedures
3. After the English answer, add a simple Kiswahili summary with ONLY key points
4. For Kiswahili: Use simple vocabulary, translate ONLY the main idea (1-2 sentences maximum)
5. Do NOT attempt full Kiswahili translation - just the essential point

IMPORTANT:
- SHIF = Social Health Insurance Fund (Kenya's new health insurance)
- If you cannot translate well to Kiswahili, skip it and just provide English
- Do not make up Kiswahili words - use English terms if needed

Context from official documents:
{context}

Question: {question}

Answer (English first, then brief Kiswahili summary if possible):"""

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "question"], 
    template=template
)

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
        
        # LCEL Chain
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
                    status.write("📖 Searching documents...")
                    source_docs = retriever.invoke(prompt)
                    
                    status.write("🤖 Generating answer...")
                    response = rag_chain.invoke(prompt)
                    
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
                    st.error(f"Error: {str(e)}")
                        
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
    st.caption("For best results, ask specific questions like:")
    st.code("""
- How do I register for SHIF?
- What are SHIF contributions?
- Who is eligible for SHIF?
- What does Article 43 say?
    """, language="text")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("🇰🇪 NIRU 2026 AI Challenge")
    st.caption("Ismail Katoya Ali")
    st.caption("HAI-2026-007")
