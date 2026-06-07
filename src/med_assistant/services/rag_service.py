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
from med_assistant.core.observability import metrics_registry
from med_assistant.services.llm_service import get_llm
from med_assistant.services.evaluation_service import EvaluatorService
from langchain.prompts import PromptTemplate

_CONVERSATIONAL_EXACT = frozenset({
    "hi", "hii", "hiii", "hello", "hey", "heya", "hiya", "howdy", "yo", "sup",
    "good morning", "good afternoon", "good evening", "good night",
    "thanks", "thank you", "thx", "ty", "thankyou",
    "bye", "goodbye", "see you", "cya", "ok", "okay", "cool", "nice",
})
_MEDICAL_HINTS = re.compile(
    r"\b(what|how|why|when|where|which|explain|symptom|treatment|disease|diagnosis|"
    r"drug|medicine|patient|dose|dosage|anemia|influenza|cancer|diabetes|infection|"
    r"therapy|clinical|guideline|condition|disorder|syndrome|pathology)\b",
    re.I,
)

_GREETING_RE = re.compile(
    r"^(hi+|hello+|hey+|hiya|howdy|yo|sup|good\s*(morning|afternoon|evening|night)|"
    r"what'?s\s*up|greetings?)[\s!.?]*$",
    re.I,
)
_THANKS_RE = re.compile(r"^(thanks?|thank\s*you|thx|ty|thankyou|appreciate\s*it)[\s!.?]*$", re.I)
_BYE_RE = re.compile(r"^(bye+|good\s*bye|see\s*ya|cya|take\s*care)[\s!.?]*$", re.I)


def is_conversational_query(question: str) -> bool:
    """Greetings, thanks, bye, and other non-document small talk — skip full RAG."""
    q = question.strip()
    if not q:
        return True

    q_lower = q.lower()
    q_clean = re.sub(r"[^\w\s]", "", q_lower).strip()
    q_norm = re.sub(r"\s+", " ", q_clean)

    if q_norm in _CONVERSATIONAL_EXACT:
        return True
    if _GREETING_RE.match(q_norm) or _THANKS_RE.match(q_norm) or _BYE_RE.match(q_norm):
        return True

    if "?" in q or _MEDICAL_HINTS.search(q):
        return False

    return len(q_norm.split()) <= 4


def _clean_generated_answer(text: str) -> str:
    """Normalize LLM output and remove stutter-style repetition from small models."""
    if not text:
        return text

    cleaned = text.strip()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Tiny models often restart the same passage mid-answer (sometimes multiple times).
    for _ in range(3):
        anchor_len = min(32, max(16, len(cleaned) // 6))
        if len(cleaned) <= anchor_len * 2:
            break
        anchor = cleaned[:anchor_len]
        repeat_at = cleaned.find(anchor, anchor_len)
        if repeat_at == -1:
            break
        first = cleaned[:repeat_at].strip()
        second = cleaned[repeat_at:].strip()
        cleaned = second if len(second) > len(first) else first

    if "\n\n" not in cleaned and len(cleaned) > 280:
        cleaned = re.sub(r"(?<=[.!?])\s+(?=[A-Z])", "\n\n", cleaned)

    return cleaned.strip()


def conversational_reply(question: str) -> str:
    q_norm = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", question.strip().lower())).strip()
    if q_norm in {"thanks", "thank you", "thx", "ty", "thankyou"} or _THANKS_RE.match(q_norm):
        return (
            "You're welcome! Let me know if you have more questions about your uploaded medical documents."
        )
    if q_norm in {"bye", "goodbye", "see you", "cya"} or _BYE_RE.match(q_norm):
        return "Goodbye! Come back anytime you need help with your medical documents."
    return (
        "Hello! I'm MedAssist, your medical document assistant. "
        "Ask me anything about conditions, symptoms, or treatments in your uploaded PDFs."
    )


class RAGService:
    def __init__(self):
        self.llm = None
        self.vectordb = None
        self.evaluator = None
        self._embeddings = None
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[Document] = []
        self._reranker: Optional[CrossEncoder] = None
        self._reranker_load_attempted: bool = False

        # Simple in-memory caches (process-local)
        self._answer_cache: Dict[str, Dict[str, Any]] = {}
        self._retrieval_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_max_entries: int = 256

    def initialize(self):
        """
        Initializes the Model and RAG chain. 
        Uses a local open-source HuggingFace model.
        """
        # 1. Get LLM (local HuggingFace) — optional defer for faster API startup
        if settings.LAZY_LLM_LOAD:
            self.llm = None
            print("LLM load deferred until first query (LAZY_LLM_LOAD=true).")
        else:
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

        self._embeddings = embeddings
        self.evaluator = None
        if settings.ENABLE_RAG_EVALUATION:
            print("Ragas evaluation available — users can enable it per query in the UI.")
        else:
            print("Ragas evaluation disabled on server (ENABLE_RAG_EVALUATION=false).")

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
6. Use Markdown to structure your answer: headers, bullet points, **bold** for key terms, *italics* for clinical emphasis, and ==double equals== around the most important findings (e.g. ==aplastic anemia==).

Context:
{context}

Question: 
{question}

Detailed Evidence-Based Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        self._qa_prompt = QA_CHAIN_PROMPT

        # Build BM25 index from persisted Chroma documents (keyword/hybrid retrieval)
        self._build_bm25_index()

        print("RAG Service initialized.")

    def _get_reranker(self) -> Optional[CrossEncoder]:
        """Load the cross-encoder lazily so startup is not blocked."""
        if self._reranker is not None:
            return self._reranker
        if self._reranker_load_attempted:
            return None
        self._reranker_load_attempted = True

        if not torch.cuda.is_available():
            print("Skipping reranker on CPU (hybrid retrieval still active).")
            return None

        try:
            print(f"Loading reranker {settings.RERANKER_MODEL_ID}...")
            self._reranker = CrossEncoder(settings.RERANKER_MODEL_ID)
        except Exception as e:
            print(f"Warning: failed to load reranker {settings.RERANKER_MODEL_ID}: {e}")
            self._reranker = None
        return self._reranker

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

    def _ensure_llm(self):
        if self.llm is None:
            print("Loading Local HuggingFace LLM (first query)...")
            self.llm = get_llm()
        return self.llm

    def _should_evaluate(self, enable_evaluation: bool) -> bool:
        return bool(settings.ENABLE_RAG_EVALUATION and enable_evaluation)

    def _ensure_evaluator(self) -> Optional[EvaluatorService]:
        if not settings.ENABLE_RAG_EVALUATION or not self._embeddings:
            return None
        if self.evaluator is None:
            print("Loading Ragas evaluator (first evaluation request)...")
            self.evaluator = EvaluatorService(self.llm, self._embeddings)
        return self.evaluator

    def _default_eval_scores(self) -> Dict[str, float]:
        return {"faithfulness": 0.0, "relevance": 0.0, "confidence_score": 0.0}

    def _evaluate_if_enabled(
        self,
        question: str,
        context: str,
        answer: str,
        enable_evaluation: bool,
    ) -> Dict[str, float]:
        if not self._should_evaluate(enable_evaluation):
            return self._default_eval_scores()
        evaluator = self._ensure_evaluator()
        if not evaluator:
            return self._default_eval_scores()
        return evaluator.evaluate_response(question, context, answer)

    def _build_result(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        eval_results: Dict[str, float],
        enable_evaluation: bool,
        obs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        enabled = self._should_evaluate(enable_evaluation)
        if not enabled:
            eval_results = self._default_eval_scores()
        result = {
            "answer": answer,
            "sources": sources,
            "confidence": eval_results["confidence_score"],
            "metrics": {
                "faithfulness": eval_results["faithfulness"],
                "relevance": eval_results["relevance"],
            },
            "evaluation_enabled": enabled,
        }
        if obs is not None:
            obs["faithfulness"] = eval_results["faithfulness"]
            obs["relevance"] = eval_results["relevance"]
            obs["evaluation_enabled"] = enabled
            result["_obs"] = obs
            metrics_registry.record_rag_query(obs)
        return result

    def _new_obs(self) -> Dict[str, Any]:
        return {
            "stages_ms": {"rewrite": 0.0, "retrieve": 0.0, "llm": 0.0, "eval": 0.0},
            "conversational": False,
            "cache_hit": False,
            "weak_retrieval": False,
            "retrieval_hit": False,
            "best_dense_distance": None,
            "source_count": 0,
        }

    def _finalize_obs(self, obs: Dict[str, Any], started_at: float) -> Dict[str, Any]:
        obs["total_ms"] = round((time() - started_at) * 1000, 2)
        return obs

    def _normalize_cached_result(
        self,
        result: Dict[str, Any],
        enable_evaluation: bool,
    ) -> Dict[str, Any]:
        """Re-apply per-request evaluation preference to in-memory cached answers."""
        if result.get("evaluation_enabled") == self._should_evaluate(enable_evaluation):
            return result
        normalized = dict(result)
        enabled = self._should_evaluate(enable_evaluation)
        normalized["evaluation_enabled"] = enabled
        if not enabled:
            normalized["confidence"] = 0.0
            normalized["metrics"] = {"faithfulness": 0.0, "relevance": 0.0}
        return normalized

    def _conversational_answer(self, question: str, enable_evaluation: bool) -> Dict[str, Any]:
        answer = conversational_reply(question)
        return self._build_result(answer, [], self._default_eval_scores(), enable_evaluation)

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
            rewritten = self._ensure_llm().invoke(prompt)
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
        reranker = self._get_reranker()
        if not reranker:
            return [(d, 0.0) for d in docs[:top_n]]
        pairs = [(query, d.page_content) for d in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        return [(d, float(s)) for d, s in ranked[:top_n]]

    def answer_question(
        self,
        question: str,
        chat_history: list = None,
        enable_evaluation: bool = False,
    ):
        if not self.vectordb:
            raise RuntimeError("RAG Service is not initialized.")
        if not settings.LAZY_LLM_LOAD and not self.llm:
            raise RuntimeError("RAG Service is not initialized.")

        started_at = time()
        obs = self._new_obs()

        if is_conversational_query(question):
            obs["conversational"] = True
            cache_key = f"conv::{question.strip().lower()}::eval={int(enable_evaluation)}"
            cached = self._answer_cache.get(cache_key)
            if cached:
                obs["cache_hit"] = True
                result = self._normalize_cached_result(cached, enable_evaluation)
                return self._build_result(
                    result["answer"],
                    result.get("sources", []),
                    {
                        "faithfulness": result.get("metrics", {}).get("faithfulness", 0.0),
                        "relevance": result.get("metrics", {}).get("relevance", 0.0),
                        "confidence_score": result.get("confidence", 0.0),
                    },
                    enable_evaluation,
                    self._finalize_obs(obs, started_at),
                )
            result = self._conversational_answer(question, enable_evaluation)
            self._answer_cache[cache_key] = {k: v for k, v in result.items() if k != "_obs"}
            return self._build_result(
                result["answer"],
                result.get("sources", []),
                {
                    "faithfulness": result["metrics"]["faithfulness"],
                    "relevance": result["metrics"]["relevance"],
                    "confidence_score": result["confidence"],
                },
                enable_evaluation,
                self._finalize_obs(obs, started_at),
            )

        rewrite_started = time()
        standalone_q = self._rewrite_question(question, chat_history)
        obs["stages_ms"]["rewrite"] = round((time() - rewrite_started) * 1000, 2)

        cache_key = f"q::{standalone_q}::eval={int(enable_evaluation)}"
        cached = self._answer_cache.get(cache_key)
        if cached:
            obs["cache_hit"] = True
            result = self._normalize_cached_result(cached, enable_evaluation)
            obs["retrieval_hit"] = bool(result.get("sources"))
            obs["source_count"] = len(result.get("sources", []))
            return self._build_result(
                result["answer"],
                result.get("sources", []),
                {
                    "faithfulness": result.get("metrics", {}).get("faithfulness", 0.0),
                    "relevance": result.get("metrics", {}).get("relevance", 0.0),
                    "confidence_score": result.get("confidence", 0.0),
                },
                enable_evaluation,
                self._finalize_obs(obs, started_at),
            )

        retrieve_started = time()
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
        obs["stages_ms"]["retrieve"] = round((time() - retrieve_started) * 1000, 2)
        obs["source_count"] = len(source_docs)

        best_dense_dist = None
        if dense_hits:
            best_dense_dist = float(sorted(dense_hits, key=lambda x: x[1])[0][1])
        obs["best_dense_distance"] = best_dense_dist

        if best_dense_dist is not None and best_dense_dist > settings.RETRIEVAL_MAX_DISTANCE:
            obs["weak_retrieval"] = True
            answer = (
                "I couldn’t find strong enough evidence in the uploaded documents to answer this reliably.\n\n"
                "Try:\n"
                "- Asking a more specific question (drug name, dosage, condition, guideline name)\n"
                "- Uploading the relevant document/PDF section\n"
                "- Providing the page/section you want me to use\n"
            )
            sources = [{"page_content": d.page_content, "metadata": d.metadata} for d in source_docs]
            eval_started = time()
            eval_results = self._evaluate_if_enabled(
                question,
                "\n".join([d.page_content for d in source_docs])[:6000],
                answer,
                enable_evaluation,
            )
            obs["stages_ms"]["eval"] = round((time() - eval_started) * 1000, 2)
            return self._build_result(
                answer, sources, eval_results, enable_evaluation, self._finalize_obs(obs, started_at)
            )

        obs["retrieval_hit"] = len(source_docs) > 0

        context = "\n\n".join(
            [
                f"[{(d.metadata or {}).get('source','unknown')} p.{(d.metadata or {}).get('page_number', (d.metadata or {}).get('page', 0) + 1)}]\n{d.page_content}"
                for d in source_docs
            ]
        )

        prompt_text = self._qa_prompt.format(context=context, question=standalone_q)
        try:
            llm_started = time()
            answer = self._ensure_llm().invoke(prompt_text)
            answer = answer.content if hasattr(answer, "content") else str(answer)
            answer = _clean_generated_answer(str(answer))
            obs["stages_ms"]["llm"] = round((time() - llm_started) * 1000, 2)
        except Exception as e:
            metrics_registry.record_error(event="rag.llm.failed", error=str(e))
            return self._build_result(
                f"I encountered an error while processing your request: {str(e)}",
                [],
                self._default_eval_scores(),
                enable_evaluation,
                self._finalize_obs(obs, started_at),
            )

        sources = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in source_docs]
        context_str = "\n".join([doc.page_content for doc in source_docs])
        if len(context_str) > 6000:
            context_str = context_str[:6000] + "... [context truncated]"

        eval_started = time()
        eval_results = self._evaluate_if_enabled(question, context_str, answer, enable_evaluation)
        obs["stages_ms"]["eval"] = round((time() - eval_started) * 1000, 2)

        result = self._build_result(
            answer, sources, eval_results, enable_evaluation, self._finalize_obs(obs, started_at)
        )

        cache_payload = {k: v for k, v in result.items() if k != "_obs"}
        self._answer_cache[cache_key] = cache_payload
        if len(self._answer_cache) > self._cache_max_entries:
            self._answer_cache.pop(next(iter(self._answer_cache)))

        return result

    def answer_question_stream(
        self,
        question: str,
        chat_history: list = None,
        enable_evaluation: bool = False,
        chunk_size: int = 48,
    ) -> Iterable[Dict[str, Any]]:
        """
        Streams an answer end-to-end. This is a lightweight stream that yields partial text chunks
        (useful for UX) plus a final summary payload.
        """
        start = time()
        result = self.answer_question(question, chat_history, enable_evaluation)
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        confidence = result.get("confidence", 0.0)
        metrics = result.get("metrics", {})
        evaluation_enabled = result.get("evaluation_enabled", False)

        yield {
            "type": "meta",
            "sources": sources,
            "confidence": confidence,
            "metrics": metrics,
            "evaluation_enabled": evaluation_enabled,
        }

        if answer:
            yield {"type": "delta", "text": answer}

        yield {"type": "done", "total_time": f"{round(time() - start, 3)} sec"}
