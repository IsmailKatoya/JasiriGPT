# JasiriGPT System Architecture

## High-Level Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    SOVEREIGN BOUNDARY                        │
│              (No Data Leaves This Environment)               │
│                                                              │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │   Streamlit  │────────▶│  app.py      │                 │
│  │   Frontend   │         │  (Orchestrator)                 │
│  └──────────────┘         └──────┬───────┘                 │
│                                   │                          │
│                                   ▼                          │
│                          ┌────────────────┐                 │
│                          │  prompts.py    │                 │
│                          │  (Anti-Halluc) │                 │
│                          └────────┬───────┘                 │
│                                   │                          │
│                     ┌─────────────┴─────────────┐           │
│                     ▼                           ▼           │
│            ┌─────────────────┐        ┌──────────────┐     │
│            │ FAISS Vectorstore│        │ Mistral 7B   │     │
│            │  (Indexed Docs)  │        │  (Offline)   │     │
│            └─────────────────┘        └──────────────┘     │
│                     ▲                                        │
│                     │                                        │
│            ┌────────┴────────┐                              │
│            │  ingest.py      │                              │
│            │  (Doc Pipeline) │                              │
│            └────────┬────────┘                              │
│                     │                                        │
│                     ▼                                        │
│            ┌─────────────────┐                              │
│            │  data/          │                              │
│            │  - Finance Act  │                              │
│            │  - SHIF Regs    │                              │
│            │  - Constitution │                              │
│            └─────────────────┘                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                  100% LOCAL PROCESSING
                  HF_HUB_OFFLINE = "1"
```

## Security Layers

### Layer 1: Data Ingestion (Verified Sources Only)
- ✅ Only official government PDFs accepted
- ✅ Manual verification before indexing
- ✅ Source tracking in metadata

### Layer 2: Vectorstore (Closed-Loop Retrieval)
- ✅ FAISS local database (no cloud sync)
- ✅ e5-base-v2 embeddings (offline mode)
- ✅ k=2 retrieval (top 2 relevant chunks only)

### Layer 3: LLM Inference (Anti-Hallucination)
- ✅ Temperature = 0 (deterministic)
- ✅ Context limited to 700 chars/chunk
- ✅ Prompt enforces "use only context"
- ✅ No internet access during generation

### Layer 4: Response Validation
- ✅ Source attribution required
- ✅ Token limit prevents rambling
- ✅ UI displays document references

---

## Data Flow Example

**Query**: "How do I register for SHIF?"

1. **User Input** → Streamlit chat interface
2. **Retrieval** → FAISS searches 3 indexed PDFs
3. **Top-2 Selection** → Returns 2 most relevant chunks
4. **Context Limiting** → Crops to 700 chars each
5. **LLM Generation** → Mistral processes with temp=0
6. **Source Addition** → Appends "Sources: SHIF_Regulations_2024.pdf"
7. **Response** → Displays 8-step registration process

**Time**: 5-10 minutes (CPU), Expected 5-10 seconds (with GPU)

---

## GPU Acceleration Plan

### Current: Intel CPU
- Inference: 5-10 min/query
- Bottleneck: Transformer matrix operations

### With NVIDIA T4:
- Inference: 5-10 sec/query
- Method: CUDA acceleration for Ollama
- Maintains all security protocols
