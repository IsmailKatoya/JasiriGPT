import streamlit as st
import os
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

# --- INITIALIZE COMPONENTS WITH TIMEOUT ---
@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    llm = ChatOllama(
        model="mistral",
        temperature=0,
        num_predict=512,  # Limit output tokens
        timeout=30  # 30 second timeout
    )
    return embeddings, llm

embeddings, llm = load_resources()

# --- SHORTER, MORE EFFICIENT PROMPT ---
template = """You are JasiriGPT, a Kenyan Government Policy Assistant.

Using the context below, answer the question. If you don't know, say so clearly.

Provide:
1. Answer in English (2-3 sentences)
2. Key point in Kiswahili (1 sentence)
3. Cite source document

Context: {context}

Question: {question}

Answer:"""

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
        
        # Create retriever with FEWER documents (k=2 instead of 3)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # Simplified document formatting
        def format_docs(docs):
            # Limit context to 1000 chars to speed up processing
            context = "\n\n".join(doc.page_content[:500] for doc in docs)
            return context[:1000]
        
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
        
        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User Input
        if prompt := st.chat_input("Ask about Finance Act 2024, SHIF, or Constitution..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                # Add status updates
                status_placeholder = st.empty()
                
                try:
                    status_placeholder.info("🔍 Retrieving documents...")
                    source_docs = retriever.get_relevant_documents(prompt)
                    
                    status_placeholder.info("🤖 Generating response...")
                    response = rag_chain.invoke(prompt)
                    
                    status_placeholder.empty()
                    
                    # Get sources
                    sources = set([
                        doc.metadata.get('source', 'Unknown').split('/')[-1] 
                        for doc in source_docs
                    ])
                    source_text = f"\n\n📄 **Sources:** {', '.join(sources)}"
                    
                    full_response = response + source_text
                    st.markdown(full_response)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response
                    })
                    
                except Exception as e:
                    status_placeholder.empty()
                    st.error(f"❌ Error: {str(e)}")
                    st.warning("⚠️ Try a simpler question or check if Ollama is running properly.")
                        
    except Exception as e:
        st.error(f"❌ Error loading the database: {e}")
else:
    st.warning("⚠️ Vectorstore not found! Please run 'python ingest.py' first.")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.markdown("### 🛡️ About JasiriGPT")
    st.info("100% Local AI - No external APIs")
    
    st.markdown("### ⚙️ System Status")
    
    # Check Ollama
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        if 'mistral' in result.stdout:
            st.success("✅ Mistral model loaded")
        else:
            st.error("❌ Mistral model not found")
            st.code("ollama pull mistral", language="bash")
    except:
        st.warning("⚠️ Cannot check Ollama status")
    
    st.markdown("### 🧪 Quick Tests")
    st.code("""
1. What is SHIF?
2. List tax changes in Finance Act
3. Eleza affordable housing
    """, language="text")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.caption("🇰🇪 NIRU 2026 | Mistral + FAISS")
