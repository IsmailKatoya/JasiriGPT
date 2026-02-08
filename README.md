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

## 📁 Project Structure
- `app.py`: Streamlit interface with real-time streaming responses.
- `prompts.py`: Modularized bilingual ChatML templates for high accuracy.
- `ingest.py`: Pipeline for indexing PDF policy documents (e.g., Finance Act, SHIF).
- `requirements.txt`: Project dependency manifest.
- `vectorstore/`: Local FAISS database for instant retrieval.

## 🛡️ National Security & Transparency Alignment
JasiriGPT supports the Kenyan National Security pillar by:
1.  **Preventing Hallucinations:** Using a strict RAG pipeline that cites official Gazette sources.
2.  **Data Sovereignty:** Local indexing on Kenyan-controlled hardware prevents external data harvesting.
3.  **Inclusive Civic Tech:** Multilingual access (English/Swahili) to reduce social misinformation.

---

## 📊 Milestone Status: Phase 1 Complete ✅
* **Architecture:** Modular design with decoupled prompt logic.
* **Performance:** Optimized query time (~8s) using streaming output.
* **Accuracy:** 90%+ factual grounding on verified SHIF and Finance Act documents.
* **Date:** February 8, 2026

---
**Developed for the NIRU AI Hackathon 2026** **By Ismail Katoya Ali**
