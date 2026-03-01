# Meeting with Nova - February 17, 2026

## Agenda Items to Discuss

### 1. Milestone Review

**Milestone 1: Environment Setup ✅ COMPLETE**
- Local development environment (Ubuntu 24.04 LTS)
- Ollama + Mistral 7B installed
- Python environment configured
- Core ingestion engine (ingest.py) operational

**Evidence**: 
- Git commit history (39 commits)
- Working system demo available
- All dependencies in requirements.txt

---

**Milestone 2: Technical Roadmap & Security ✅ DOCUMENTED**
- System architecture defined (see ARCHITECTURE.md)
- Anti-hallucination protocol implemented (see SECURITY_PROTOCOL.md)
- Offline mode enforced (HF_HUB_OFFLINE="1")
- Closed-loop RAG with source verification

**Evidence**:
- SECURITY_PROTOCOL.md (comprehensive documentation)
- app.py code showing offline enforcement
- Testing results showing hallucination prevention

---

**Milestone 3: Bilingual Policy Retrieval ⚠️ IN PROGRESS**
- English responses: Working accurately
- Swahili responses: Disabled due to hallucination issues
- Alpha UI: Functional Streamlit interface
- Policy retrieval: Operational for SHIF, Constitution

**Challenges**:
- Mistral 7B not optimized for Kiswahili
- Need specialized Swahili model or fine-tuning
- Current CPU limits response time (5-10 min)

---

### 2. Current Challenges

**Challenge 1: Response Time**
- Current: 5-10 minutes per query (CPU-bound)
- Impact: Not viable for real-time citizen engagement
- Solution: NVIDIA T4 GPU request pending

**Challenge 2: Kiswahili Quality**
- Current: Hallucinations in Swahili translation
- Impact: Disabled Swahili to maintain accuracy
- Solution: Need NLLB model or fine-tuned Swahili LLM

**Challenge 3: SHIF Definition Retrieval**
- Current: Retrieves emergency care section instead of main definition
- Impact: Incorrect answer to "What is SHIF?"
- Solution: Better chunking strategy or synthetic FAQ

---

### 3. Questions for Nova

1. **GPU Access**: Timeline for T4 GPU approval? This is critical for Milestone 3.

2. **Swahili Model**: Recommendations for Kenyan Swahili LLM? Options:
   - Fine-tune Mistral on Swahili corpus
   - Use NLLB-200 translation layer
   - Partner with local NLP research group

3. **Milestone Approval**: What additional evidence needed for Milestones 1 & 2?

4. **Document Sources**: Can NIRU provide verified government document corpus?

5. **Testing Phase**: When can we begin user testing with real citizens?

---

### 4. Demonstrations Available

✅ Live system demo (streamlit run app.py)
✅ Code walkthrough (app.py, prompts.py, ingest.py)
✅ Git history (39 commits showing progress)
✅ Security documentation (SECURITY_PROTOCOL.md)
✅ Architecture diagram (ARCHITECTURE.md)

---

### 5. Next Steps Post-Meeting

**If Milestones 1 & 2 Approved**:
- Focus on Milestone 3 optimization
- Implement GPU acceleration when available
- Improve Swahili translation pipeline

**If Additional Work Needed**:
- Address Nova's feedback immediately
- Resubmit within 48 hours
- Schedule follow-up review

---

**Meeting Time**: February 17, 2026, 07:00 PM (19:00 hours)  
**Mentor**: Nova  
**Project**: JasiriGPT (HAI-2026-007)  
**Developer**: Ismail Katoya Ali
