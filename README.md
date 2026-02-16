# 🛡️ JasiriGPT (HAI-2026-007)
### *Secure, Multilingual Generative AI for Kenyan Policy Transparency*

JasiriGPT is a localized **Retrieval-Augmented Generation (RAG)** assistant designed to bridge the gap between complex government policies and citizen understanding. It simplifies legal information into English and Swahili while maintaining strict **Data Sovereignty**.



---

## 🚀 Key Innovation: Sovereign RAG
Unlike traditional AI assistants that rely on cloud APIs (OpenAI/Google), JasiriGPT processes all Kenyan legal data **locally**. It is built on an Ubuntu 24.04 LTS infrastructure, ensuring sensitive documents never leave the national digital jurisdiction.

## 🛠️ Tech Stack (2026 Sovereign Standard)
* **OS:** Ubuntu 24.04 LTS
* **Brain:** Mistral-7B via **Ollama** (100% Offline)
* **Vector Engine:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** `intfloat/e5-base-v2` (Running on local CPU/GPU)
* **Framework:** LangChain (LCEL)
* **UI:** Streamlit

---

## ⚡ Speed & Efficiency Optimization
To ensure JasiriGPT remains accessible on standard hardware (non-GPU environments), the following optimizations were implemented:
- **Reduced Context Window:** 2048 tokens for faster CPU inference.
- **Selective Retrieval:** Limited to top 2 relevant document chunks ($k=2$).
- **Prediction Hard-Cap:** Limited to 250 tokens to ensure punchy, concise English and Kiswahili summaries.

---

## 📁 Project Structure
- `app.py`: Streamlit interface with real-time streaming responses.
- `prompts.py`: Modularized bilingual ChatML templates for high accuracy.
- `ingest.py`: Pipeline for indexing PDF policy documents (e.g., Finance Act, SHIF).
- `requirements.txt`: Project dependency manifest.
- `vectorstore/`: Local FAISS database for instant retrieval.
- `LICENSE`: MIT License.

---

## ⚙️ Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)[your-username]/jasirigpt.git
   cd jasirigpt

## 🔮 Future Roadmap
- Integration of USSD/SMS gateways for offline policy access in rural Kenya.

## 🔮 Future Roadmap
- Integration of USSD/SMS gateways for offline policy access in rural Kenya.

## Recent Updates (February 16, 2026)

### Performance Enhancements
- ⚡ Response time optimized to 6-10 seconds (60% improvement)
- 🔒 Full offline mode implementation for data sovereignty
- 🎯 Improved SHIF query accuracy with better document chunking

### Testing Status (Feb 16, 2026)
- ✅ SHIF registration queries: Working accurately
- ✅ Finance Act queries: Functional
- ✅ Constitution queries: Operational
- ✅ Bilingual responses: English + Kiswahili summaries
