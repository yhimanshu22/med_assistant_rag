"""Helpers to exercise the RAG pipeline without loading LLMs, embeddings, or ChromaDB."""

from __future__ import annotations

from typing import Any, Callable, List, Tuple
from unittest.mock import MagicMock

from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from med_assistant.services.rag_service import RAGService

QA_PROMPT_TEMPLATE = """
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


def make_grounded_llm(expected_terms: List[str] | None = None) -> MagicMock:
    """Mock LLM that returns an answer echoing terms found in the prompt context."""

    def invoke(prompt: str) -> MagicMock:
        response = MagicMock()
        prompt_lower = prompt.lower()
        if expected_terms:
            matched = [term for term in expected_terms if term.lower() in prompt_lower]
        else:
            matched = []
        if matched:
            response.content = (
                "Based on the provided medical documents, "
                + ", ".join(matched)
                + " are described in the retrieved context."
            )
        else:
            response.content = "I do not have enough specific information from the provided documents."
        return response

    llm = MagicMock()
    llm.invoke = invoke
    return llm


def make_unit_rag_service(
    llm: MagicMock | None = None,
    llm_factory: Callable[[], MagicMock] | None = None,
) -> RAGService:
    service = RAGService()
    service.llm = llm or (llm_factory() if llm_factory else make_grounded_llm())
    service.vectordb = MagicMock()
    service._qa_prompt = PromptTemplate.from_template(QA_PROMPT_TEMPLATE)
    return service


def patch_retrieval(
    service: RAGService,
    primary_chunk: Document,
    dense_distance: float,
    extra_chunks: List[Document] | None = None,
) -> None:
    """Stub dense/BM25 retrieval so tests hit a known context without a vector DB."""
    chunks = [primary_chunk] + (extra_chunks or [])

    def dense_retrieve(query: str, k: int) -> List[Tuple[Document, float]]:
        hits = [(chunks[0], dense_distance)]
        for doc in chunks[1:k]:
            hits.append((doc, dense_distance + 0.1))
        return hits[:k]

    def bm25_retrieve(query: str, k: int) -> List[Tuple[Document, float]]:
        return [(doc, 1.0 - i * 0.05) for i, doc in enumerate(chunks[:k])]

    service._dense_retrieve = dense_retrieve  # type: ignore[method-assign]
    service._bm25_retrieve = bm25_retrieve  # type: ignore[method-assign]
    service._get_reranker = lambda: None  # type: ignore[method-assign, return-value]


def run_sample_case(service: RAGService, case: dict[str, Any]) -> dict[str, Any]:
    patch_retrieval(
        service,
        primary_chunk=case["primary_chunk"],
        dense_distance=case["dense_distance"],
    )
    service.llm = make_grounded_llm(case.get("expected_terms", []))
    return service.answer_question(case["question"])
