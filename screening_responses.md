# Technical Screening Responses

### Screening Qn1: Describe a project where you had to balance model performance with real-world constraints (latency, cost, device limits, safety, or data scarcity). What trade-offs did you make?
**Response:**
In a recent Medical Assistant RAG project, I was constrained by hardware limits: the application needed to run locally on consumer-grade CPUs without relying on expensive, cloud-hosted LLM APIs. This required deploying a 1.1 Billion parameter model (TinyLlama) using HuggingFace Transformers. Initially, the latency was severe, taking up to 185 seconds to generate a response because the model evaluated too much retrieved context without hardware acceleration.

To balance performance and usability, I made a significant trade-off in the RAG retrieval strategy. I reduced the context chunk size from 1,000 characters down to 500, dropping the total evaluated context volume by half, and casting the model calculations to `bfloat16`. The trade-off was that providing less background text slightly limited the AI's semantic breadth, but it drastically reduced inference times to a matter of seconds, making the system actually usable for real-time interactions.

### Screening Qn2: How would you evaluate and reduce hallucinations in an LLM used for learning content?
**Response:**
To evaluate hallucinations, I closely monitor the generated text for repetitive loops or fabricated facts that run tangential to the provided context. For instance, in my Medical Assistant project, the LLM hallucinatively began generating endless hypothetical "Question/Answer" pairs instead of simply answering the user's prompt. 

To reduce this, I employed strict physical and prompt-based constraints. I capped the generation window (`max_new_tokens=256`) to literally cut off runaway hallucinatory loops. I introduced a `repetition_penalty` to the inference pipeline to mathematically discourage the model from reusing identical sequences. Finally, I hardened the prompt engineering to explicitly command the model: "If unknown, say you do not know." rather than letting it guess. Setting up localized, highly dense RAG chunks also means the model is less likely to extrapolate incorrectly because the provided context is strictly relevant to the query.

### Screening Qn3: Give us an example of prompting techniques you have used. How did you evaluate the prompt's effectiveness in improving response quality or reliability?
**Response:**
I frequently use role-playing and constraint-based zero-shot prompting. For example, instead of a vague "Answer the question based on the text," I instruct the model: "You are a concise medical assistant. Use the context to answer the question briefly. If unknown, say you do not know."

I evaluated its effectiveness through empirical observation of the output latency and structure. Before applying the concise constraints, the model would attempt to generate highly elaborate answers, failing to recognize when it had fundamentally answered the question, which triggered generation timeouts. By forcing brevity and an "out route" (admitting ignorance), the responses became highly reliable; the model stopped trying to invent text to fill space, dramatically improving both the accuracy of the medical information retrieved and the reliability of the system's response time.

### Screening Qn4: Architectural Design (RAG & Data) Context
*Point 1: Describe the architecture of a RAG system you've built (specifically your choice of vector database and chunking strategy).*
*Point 2: What was the biggest challenge you faced in making the retrieval relevant to the user's query?*
*Requirement: Upload a screenshot of your pipeline logic or a database schema.*
**Response:**
The architecture of my offline RAG system utilizes a FastAPI backend and a Streamlit frontend. For the vector database, I chose **ChromaDB** because it is lightweight, open-source, and persists locally via SQLite, avoiding the need for external database hosting. My chunking strategy utilizes Langchain's `RecursiveCharacterTextSplitter`. I optimized the chunk size to 500 characters with a 50-character overlap, allowing for highly dense, semantically focused context blocks.

The biggest challenge was maintaining relevance while strictly limiting the retrieval volume to save CPU inference time. Because I could only afford to pass a few chunks into the LLM, any irrelevant text would derail the answer. I solved this by shrinking the chunk size from 1000 down to 500 characters and retrieving 3 focused chunks (`k=3`) rather than 2 large, noisy chunks.

*(Note for upload: Take a screenshot of the `ingestion_service.py` where the `RecursiveCharacterTextSplitter` and `Chroma` database are initialized.)*

### Screening Qn5: Operational Constraints & Trade-offs
*Point 1: Share a project where you had to optimize for a specific constraint (like latency, API costs, or limited GPU memory).*
*Point 2: What specific trade-off did you make, and how did it impact the final user experience?*
*Requirement: Upload a screenshot of your performance metrics, logs, or cost analysis.*
**Response:**
I optimized a Medical Assistant AI to operate under strict zero-API-cost constraints and zero GPU memory (a CPU-bound environment). The latency for a standard transformers pipeline in this environment was completely unmanageable.

I traded off peak model intelligence and maximum context windows for raw speed. I deployed a smaller 1.1B parameter model (`TinyLlama`), forced the precision down to `bfloat16`, disabled dynamic sampling (`do_sample=False`), and created an explicit local `models_cache` to bypass network checks on startup. For the user experience, what started as a brittle application that timed out after 3 minutes became a snappy, responsive chat interface that instantly ingested PDFs via a streaming API endpoint and answered queries in seconds. 

*(Note for upload: Take a screenshot of the `performance_logs.txt` file I created in your project root. It provides a timestamped breakdown of the 76s baseline vs the 12s optimized inference metrics, which perfectly demonstrates the trade-off success.)*

### Screening Qn6: End-to-End Development
*Point 1: Outline the tech stack you used for a deployed AI project (from the model API to the frontend/hosting).*
*Point 2: How did you handle "state" or conversation memory within the app?*
*Requirement: Upload a screenshot of the live UI or your deployment dashboard (Vercel, AWS, etc.).*
**Response:**
For my end-to-end RAG application, I used `uv` as the lightning-fast package manager. The backend is powered by **FastAPI** to serve REST endpoints (like `/query` and a streaming `/ingest`). The machine learning orchestration is handled by **Langchain** and **HuggingFace Transformers**, utilizing a local **ChromaDB** instance for vector storage. The frontend uses **Streamlit** to provide a conversational chat interface.

To handle conversation state within the typically stateless HTTP/Streamlit architecture, I leveraged Streamlit's `st.session_state`. I initialized a persistent message array that tracks the `role` (user vs. assistant) and the `content` of the conversation. Every time the user interacts with the app, the UI re-renders the entire history from that session state array, simulating a continuous, stateful memory loop directly in the browser front-end without requiring a persistent SQL database for chat logs.

*(Note for upload: Take a screenshot of the full Streamlit UI with the chat history and the sidebar PDF uploader visible.)*

---

### AI Tool Usage Breakdown
In developing these solutions and formulating responses, AI engineering tools were utilized to rapidly prototype architectural changes. Specifically, I leverage agentic AI to assist in safely refactoring monolithic codebase structures into domain-driven modular packages (`src/api`, `src/services`), migrating dependency management frameworks (`pip` to `uv`), and rapidly prototyping streaming UI endpoints. The AI acts as a sophisticated pair-programmer that helps scaffold boilerplate tests in `pytest` and flags potential memory bottlenecks in HuggingFace pipelines, which informs the strategic trade-offs discussed above.
