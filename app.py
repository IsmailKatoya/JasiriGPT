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

# --- INITIALIZE COMPONENTS ---
@st.cache_resource
def load_resources():
    try:
        # Use local embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception:
        st.warning("⚠️ Connection issue. Using local_files_only mode.")
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-base-v2",
            model_kwargs={'local_files_only': True}
        )
        
    # Mistral with Temperature 0 for strictly factual responses
    llm = ChatOllama(model="mistral", temperature=0)
    return embeddings, llm

embeddings, llm = load_resources()

# --- ANTI-HALLUCINATION BILINGUAL PROMPT ---
template = """You are JasiriGPT, the Lead Policy Analyst for the Kenyan Government.
Your goal is to provide accurate, transparent information based ONLY on the context provided.

STRICT RULES:
1. Grounding: If the answer is not in the context, say "Habari hii haipatikani kwenye hifadhi yetu." Do not make up facts.
2. Structure: 
   - First, provide a clear 2-3 sentence explanation in English.
   - Second, provide a direct Swahili (Kiswahili Sanifu) translation of the key fact. Avoid medical jargon unless it is in the text.
3. Anti-Hallucination: Before translating to Swahili, verify the fact exists in the English context.

Context: {context}

Question: {question}

Official Response:"""

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
        
        # k=3 for a balance of speed and context
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
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
                    # FIX: Use .invoke() instead of .get_relevant_documents()
                    status.write("Reading policy documents...")
                    source_docs = retriever.invoke(prompt)
                    
                    status.write("Translating and formatting...")
                    response = rag_chain.invoke(prompt)
                    
                    # Extract source names
                    sources = set([
                        doc.metadata.get('source', 'Unknown').split('/')[-1] 
                        for doc in source_docs
                    ])
                    source_text = f"\n\n📄 **Sources:** {', '.join(sources)}"
                    
                    full_response = response + source_text
                    status.update(label="✅ Response Generated", state="complete", expanded=False)
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response
                    })
                    
                except Exception as e:
                    status.update(label="❌ System Error", state="error")
                    st.error(f"Error: {str(e)}")
                        
    except Exception as e:
        st.error(f"❌ Error loading the database: {e}")
else:
    st.warning("⚠️ Vectorstore not found! Please run 'python ingest.py' first.")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.markdown("### 🛡️ JasiriGPT Status")
    st.info("Sovereign Mode: Localhost Only")
    
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()
    
    st.caption("🇰🇪 Developed for NIRU 2026")
