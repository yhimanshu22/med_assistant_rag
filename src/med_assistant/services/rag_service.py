from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import torch
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from time import time

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from med_assistant.core.config import settings
from med_assistant.services.llm_service import get_llm
from med_assistant.services.evaluation_service import EvaluatorService
from langchain.prompts import PromptTemplate

class RAGService:
    def __init__(self):
        self.llm = None
        self.vectordb = None
        self.evaluator = None # Initialize evaluator attribute
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[Document] = []
        self._reranker: Optional[CrossEncoder] = None

        # Simple in-memory caches (process-local)
        self._answer_cache: Dict[str, Dict[str, Any]] = {}
        self._retrieval_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_entries: int = 256

    def initialize(self):
        """
        Initializes the Model and RAG chain. 
        Uses a local open-source HuggingFace model.
        """
        # 1. Get LLM (local HuggingFace)
        self.llm = get_llm()


        # 2. Load Embeddings
        # Check device for embedding model
        model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL, 
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=settings.MODEL_CACHE_DIR
        )

        # 1.1 Initialize Evaluator using the selected LLM
        self.evaluator = EvaluatorService(self.llm, embeddings)

        # 3. Load VectorDB
        print(f"Loading ChromaDB from {settings.DB_DIR}...")
        self.vectordb = Chroma(persist_directory=settings.DB_DIR, embedding_function=embeddings)

        template = """
You are a highly accurate and professional Medical Assistant. 
Your goal is to provide evidence-based answers using ONLY the provided medical context.

INSTRUCTIONS:
1. Base your answer strictly on the provided context. 
2. If the context does not contain enough information to answer the question, state clearly that you do not have enough specific information from the provided documents.
3. Maintain a professional, clinical, and helpful tone.
4. If there are conflicting details in the context, mention them.
5. Do NOT hallucinate or use outside knowledge that isn't supported by the context.
6. Use Markdown (headers, bullet points, and bold text) to structure your answer for professional clarity and readability.

Context:
{context}

Question: 
{question}

Detailed Evidence-Based Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        self._qa_prompt = QA_CHAIN_PROMPT

        # Build BM25 index from persisted Chroma documents (keyword/hybrid retrieval)
        self._build_bm25_index()

        # Load cross-encoder reranker
        try:
            self._reranker = CrossEncoder(settings.RERANKER_MODEL_ID)
        except Exception as e:
            print(f"Warning: failed to load reranker {settings.RERANKER_MODEL_ID}: {e}")
            self._reranker = None

        print("RAG Service initialized.")

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _build_bm25_index(self) -> None:
        if not self.vectordb:
            return
        try:
            raw = self.vectordb.get(include=["documents", "metadatas"])
            docs: List[Document] = []
            for content, meta in zip(raw.get("documents", []), raw.get("metadatas", [])):
                if not content:
                    continue
                docs.append(Document(page_content=content, metadata=meta or {}))
            self._bm25_docs = docs
            corpus = [self._tokenize_for_bm25(d.page_content) for d in docs]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            print(f"BM25 index built with {len(docs)} chunks.")
        except Exception as e:
            print(f"Warning: failed to build BM25 index from Chroma: {e}")
            self._bm25_docs = []
            self._bm25 = None

    def _rewrite_question(self, question: str, chat_history: Optional[list]) -> str:
        if not chat_history:
            return question
        try:
            history_text = "\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in chat_history])
            prompt = (
                "Given the following conversation and a follow up question, rewrite the follow up question "
                "to be a single standalone question in the same language.\n\n"
                f"Chat History:\n{history_text}\n\n"
                f"Follow Up Input: {question}\n\nStandalone question:"
            )
            rewritten = self.llm.invoke(prompt)
            rewritten = rewritten.content if hasattr(rewritten, "content") else str(rewritten)
            rewritten = rewritten.strip()
            return rewritten if rewritten else question
        except Exception:
            return question

    def _dense_retrieve(self, query: str, k: int) -> List[Tuple[Document, float]]:
        # Returns (doc, distance). Lower distance is better.
        if not self.vectordb:
            return []
        try:
            return self.vectordb.similarity_search_with_score(query, k=k)
        except Exception:
            # Fallback if score API differs
            docs = self.vectordb.similarity_search(query, k=k)
            return [(d, 0.0) for d in docs]

    def _bm25_retrieve(self, query: str, k: int) -> List[Tuple[Document, float]]:
        if not self._bm25 or not self._bm25_docs:
            return []
        tokens = self._tokenize_for_bm25(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [(self._bm25_docs[i], float(s)) for i, s in ranked if s > 0]

    def _merge_hybrid(
        self,
        dense: List[Tuple[Document, float]],
        bm25: List[Tuple[Document, float]],
        top_n: int,
    ) -> List[Tuple[Document, Dict[str, Any]]]:
        """
        Merge dense+bm25 results by doc identity and simple rank-based fusion.
        Returns list of (doc, debug_info).
        """
        def doc_key(d: Document) -> str:
            m = d.metadata or {}
            return str(m.get("chunk_id") or f"{m.get('source')}::{m.get('page')}::{hash(d.page_content)}")

        merged: Dict[str, Dict[str, Any]] = {}

        # Dense: lower distance better → convert to rank score
        for rank, (d, dist) in enumerate(dense):
            key = doc_key(d)
            entry = merged.setdefault(key, {"doc": d, "dense_rank": None, "dense_distance": None, "bm25_rank": None, "bm25_score": None})
            entry["dense_rank"] = rank
            entry["dense_distance"] = float(dist)

        for rank, (d, score) in enumerate(bm25):
            key = doc_key(d)
            entry = merged.setdefault(key, {"doc": d, "dense_rank": None, "dense_distance": None, "bm25_rank": None, "bm25_score": None})
            entry["bm25_rank"] = rank
            entry["bm25_score"] = float(score)

        def fused(e: Dict[str, Any]) -> float:
            # Reciprocal rank fusion variant
            s = 0.0
            if e["dense_rank"] is not None:
                s += 1.0 / (50 + e["dense_rank"])
            if e["bm25_rank"] is not None:
                s += 1.0 / (50 + e["bm25_rank"])
            return s

        ranked = sorted(merged.values(), key=fused, reverse=True)[: max(top_n, 1)]
        return [(e["doc"], e) for e in ranked]

    def _rerank(self, query: str, docs: List[Document], top_n: int) -> List[Tuple[Document, float]]:
        if not docs:
            return []
        if not self._reranker:
            return [(d, 0.0) for d in docs[:top_n]]
        pairs = [(query, d.page_content) for d in docs]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        return [(d, float(s)) for d, s in ranked[:top_n]]

    def answer_question(self, question: str, chat_history: list = None):
        if not self.vectordb or not self.llm:
            raise RuntimeError("RAG Service is not initialized.")

        standalone_q = self._rewrite_question(question, chat_history)
        cache_key = f"q::{standalone_q}"
        cached = self._answer_cache.get(cache_key)
        if cached:
            return cached

        # Retrieval cache (docs only; avoids repeated DB lookups on same query)
        r_cached = self._retrieval_cache.get(cache_key)
        if r_cached:
            dense_hits = r_cached["dense_hits"]
            bm25_hits = r_cached["bm25_hits"]
            source_docs = r_cached["source_docs"]
        else:
            dense_hits = self._dense_retrieve(standalone_q, k=settings.RETRIEVE_K_DENSE)
            bm25_hits = self._bm25_retrieve(standalone_q, k=settings.RETRIEVE_K_BM25)

            merged = self._merge_hybrid(dense_hits, bm25_hits, top_n=max(settings.RETRIEVE_K_DENSE, settings.RETRIEVE_K_BM25))
            merged_docs = [d for d, _ in merged]

            reranked = self._rerank(standalone_q, merged_docs, top_n=settings.RERANK_TOP_N)
            source_docs = [d for d, _ in reranked]

            self._retrieval_cache[cache_key] = {"dense_hits": dense_hits, "bm25_hits": bm25_hits, "source_docs": source_docs}
            if len(self._retrieval_cache) > self._cache_max_entries:
                self._retrieval_cache.pop(next(iter(self._retrieval_cache)))

        # Thresholding: if dense retrieval is very weak, refuse/ask for more info
        best_dense_dist = None
        if dense_hits:
            best_dense_dist = float(sorted(dense_hits, key=lambda x: x[1])[0][1])
        if best_dense_dist is not None and best_dense_dist > settings.RETRIEVAL_MAX_DISTANCE:
            answer = (
                "I couldn’t find strong enough evidence in the uploaded documents to answer this reliably.\n\n"
                "Try:\n"
                "- Asking a more specific question (drug name, dosage, condition, guideline name)\n"
                "- Uploading the relevant document/PDF section\n"
                "- Providing the page/section you want me to use\n"
            )
            sources = [{"page_content": d.page_content, "metadata": d.metadata} for d in source_docs]
            eval_results = self.evaluator.evaluate_response(question, "\n".join([d.page_content for d in source_docs])[:6000], answer)
            return {
                "answer": answer,
                "sources": sources,
                "confidence": eval_results["confidence_score"],
                "metrics": {
                    "faithfulness": eval_results["faithfulness"],
                    "relevance": eval_results["relevance"],
                },
            }

        context = "\n\n".join(
            [
                f"[{(d.metadata or {}).get('source','unknown')} p.{(d.metadata or {}).get('page_number', (d.metadata or {}).get('page', 0) + 1)}]\n{d.page_content}"
                for d in source_docs
            ]
        )

        prompt_text = self._qa_prompt.format(context=context, question=standalone_q)
        try:
            answer = self.llm.invoke(prompt_text)
            answer = answer.content if hasattr(answer, "content") else str(answer)
            answer = str(answer).strip()
        except Exception as e:
            return {
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "metrics": {"faithfulness": 0.0, "relevance": 0.0},
            }

        # Format sources
        sources = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in source_docs]
        
        # Perform real-time evaluation for Clinical Reliability
        # Truncate context to a safe length for LLM evaluation (approx 1500 tokens / 6000 chars)
        context_str = "\n".join([doc.page_content for doc in source_docs])
        if len(context_str) > 6000:
            context_str = context_str[:6000] + "... [context truncated]"
            
        eval_results = self.evaluator.evaluate_response(question, context_str, answer)
        
        result = {
            "answer": answer,
            "sources": sources,
            "confidence": eval_results["confidence_score"],
            "metrics": {
                "faithfulness": eval_results["faithfulness"],
                "relevance": eval_results["relevance"]
            }
        }

        self._answer_cache[cache_key] = result
        if len(self._answer_cache) > self._cache_max_entries:
            self._answer_cache.pop(next(iter(self._answer_cache)))

        return result

    def answer_question_stream(self, question: str, chat_history: list = None, chunk_size: int = 48) -> Iterable[Dict[str, Any]]:
        """
        Streams an answer end-to-end. This is a lightweight stream that yields partial text chunks
        (useful for UX) plus a final summary payload.
        """
        start = time()
        result = self.answer_question(question, chat_history)
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        confidence = result.get("confidence", 1.0)
        metrics = result.get("metrics", {})

        # Initial metadata (lets frontend show citations area immediately if desired)
        yield {"type": "meta", "sources": sources, "confidence": confidence, "metrics": metrics}

        for i in range(0, len(answer), chunk_size):
            yield {"type": "delta", "text": answer[i : i + chunk_size]}

        yield {"type": "done", "total_time": f"{round(time() - start, 3)} sec"}
