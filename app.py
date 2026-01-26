import streamlit as st
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ---------------------------
# CONFIG
# ---------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-70b-versatile"   # You can change to gemma-2-9b-it

# ---------------------------
# LOAD EMBEDDINGS + FAISS
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    return db

vectorstore = load_vectorstore()

# ---------------------------
# GROQ CLIENT
# ---------------------------
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

client = get_groq_client()

# ---------------------------
# ANSWER GENERATION
# ---------------------------
def generate_answer(question, retrieved_chunks):

    context_text = "\n\n".join([f"Section {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])

    prompt = f"""
You are JasiriGPT — an expert on the Constitution of Kenya (2010).
Answer the user's question **accurately, clearly, and concisely**, using ONLY the information from the provided context.

Question:
{question}

Context (official Kenya Constitution sections):
{context_text}

Instructions:
- Give a clean, short, correct answer.
- Then list the most relevant constitutional sections.
- Do NOT add anything outside the Constitution.
    """

    llm_response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    return llm_response.choices[0].message.content


# ---------------------------
# STREAMLIT APP UI
# ---------------------------
st.title("🟩 JasiriGPT — Kenya Constitution Assistant")
st.write("Ask anything about the **Constitution of Kenya (2010)** and get precise legal answers.")

question = st.text_input("Enter your question:")

if question:
    # Step 1: Retrieve top chunks
    results = vectorstore.similarity_search(question, k=3)
    retrieved_chunks = [r.page_content for r in results]

    # Step 2: Display relevant sections
    st.subheader("📌 Top Relevant Sections")
    for r in retrieved_chunks:
        st.markdown(f"```\n{r}\n```")

    # Step 3: Generate answer
    st.subheader("🧠 JasiriGPT Answer")
    answer = generate_answer(question, retrieved_chunks)
    st.write(answer)
