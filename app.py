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
    .stChatMessage { border-radius: 10px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ JasiriGPT: Kenyan Policy Assistant")
st.subheader("Sovereign AI Prototype - NIRU 2026")

# --- INITIALIZE COMPONENTS ---
@st.cache_resource
def load_resources():
    try:
        # Optimized local embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception:
        st.warning("⚠️ High latency or offline. Using local_files_only mode.")
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-base-v2",
            model_kwargs={'local_files_only': True}
        )
        
    # Enable Streaming for better UX and temp 0 for accuracy
    llm = ChatOllama(model="mistral", temperature=0, streaming=True)
    return embeddings, llm

embeddings, llm = load_resources()

# --- ANTI-HALLUCINATION BILINGUAL PROMPT ---
template = """You are JasiriGPT, the Lead Policy Analyst for the Kenyan Government.
Your goal is to provide accurate information based ONLY on the context provided.

STRICT RULES:
1. GROUNDING: If the answer is not in the context, say "Samahani, maelezo haya hayapatikani kwenye hifadhi yetu." 
2. NO GUESSING: If the context mentions 'shifts' (time/work), do not assume it means 'SHIF' (Health Fund).
3. STRUCTURE: 
   - Start with a 2-sentence English explanation.
   - Follow with 'KWA KISWAHILI:' and a direct Swahili translation.
   - Do not translate the name 'SHIF' or 'Social Health Insurance Fund'.

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
        
        # Increased k for better coverage of complex documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # LCEL Chain for streaming
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
        
        if prompt := st.chat_input("Ask about SHIF, Finance Act, or Constitution..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                # Use a placeholder for the streaming effect
                response_placeholder = st.empty()
                full_response = ""
                
                try:
                    # Retrieve sources for citation
                    source_docs = retriever.invoke(prompt)
                    
                    # Stream the LLM response
                    for chunk in rag_chain.stream(prompt):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    
                    # Process sources
                    sources = set([
                        doc.metadata.get('source', 'Unknown').split('/')[-1] 
                        for doc in source_docs
                    ])
                    source_text = f"\n\n📄 **Sources:** {', '.join(sources)}"
                    
                    # Finalize message
                    final_output = full_response + source_text
                    response_placeholder.markdown(final_output)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": final_output
                    })
                    
                except Exception as e:
                    st.error(f"❌ System Error: {str(e)}")
                    st.warning("Make sure Ollama is running (`ollama serve`).")
                        
    except Exception as e:
        st.error(f"❌ Error loading the database: {e}")
else:
    st.warning("⚠️ Vectorstore not found! Please run 'python ingest.py' first.")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.markdown("### 🛡️ JasiriGPT Status")
    st.success("Sovereign Mode: Active")
    st.info("Mistral 7B + FAISS")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.caption("🇰🇪 NIRU 2026 Innovation Challenge")
