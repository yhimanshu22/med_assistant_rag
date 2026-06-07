"""Fast RAG pipeline tests using sample medical context — no GPU or model download."""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from med_assistant.services.rag_service import RAGService
from tests.fixtures.medical_samples import (
    APLASTIC_ANEMIA_CHUNK,
    INFLUENZA_CHUNK,
    SAMPLE_QUERY_CASES,
    WEAK_RETRIEVAL_CASE,
)
from tests.fixtures.rag_test_utils import (
    make_grounded_llm,
    make_unit_rag_service,
    patch_retrieval,
    run_sample_case,
)


@pytest.fixture
def rag_service():
    return make_unit_rag_service()


@pytest.mark.parametrize("case", SAMPLE_QUERY_CASES, ids=[c["id"] for c in SAMPLE_QUERY_CASES])
def test_sample_queries_return_context_grounded_answers(rag_service, case):
    result = run_sample_case(rag_service, case)

    answer = result["answer"].lower()
    assert result["sources"], "Expected retrieved source documents"
    assert result["sources"][0]["page_content"] == case["primary_chunk"].page_content

    for term in case["expected_terms"]:
        assert term.lower() in answer, f"Expected '{term}' in grounded answer"

    assert "strong enough evidence" not in answer


def test_weak_retrieval_returns_refusal(rag_service):
    patch_retrieval(
        rag_service,
        primary_chunk=WEAK_RETRIEVAL_CASE["primary_chunk"],
        dense_distance=WEAK_RETRIEVAL_CASE["dense_distance"],
    )

    result = rag_service.answer_question(WEAK_RETRIEVAL_CASE["question"])

    assert "strong enough evidence" in result["answer"].lower()
    assert result["sources"]


def test_prompt_includes_retrieved_context(rag_service):
    captured: dict[str, str] = {}

    def capture_invoke(prompt: str):
        captured["prompt"] = prompt
        response = MagicMock()
        response.content = "Captured context-based answer."
        return response

    rag_service.llm.invoke = capture_invoke
    patch_retrieval(rag_service, APLASTIC_ANEMIA_CHUNK, dense_distance=0.2)

    rag_service.answer_question("What is aplastic anemia?")

    assert "bone marrow" in captured["prompt"].lower()
    assert "aplastic_anemia_guide.pdf" in captured["prompt"]


def test_hybrid_merge_boosts_docs_found_by_both_retrievers():
    service = make_unit_rag_service()
    dense_doc = Document(page_content="dense only", metadata={"chunk_id": "a"})
    shared_doc = Document(page_content="shared hit", metadata={"chunk_id": "shared"})
    bm25_doc = Document(page_content="bm25 only", metadata={"chunk_id": "b"})

    merged = service._merge_hybrid(
        dense=[(shared_doc, 0.2), (dense_doc, 0.4)],
        bm25=[(shared_doc, 2.5), (bm25_doc, 1.0)],
        top_n=3,
    )

    assert merged[0][0].page_content == "shared hit"


def test_conversational_query_skips_retrieval(rag_service):
    rag_service._dense_retrieve = lambda *a, **k: pytest.fail("Should not retrieve")  # type: ignore[method-assign]

    result = rag_service.answer_question("hello")

    assert "MedAssist" in result["answer"]
    assert result["sources"] == []


def test_aplastic_anemia_sources_match_sample_context(rag_service):
    result = run_sample_case(rag_service, SAMPLE_QUERY_CASES[0])

    metadata = result["sources"][0]["metadata"]
    assert metadata["source"] == "aplastic_anemia_guide.pdf"
    assert "bone marrow" in result["sources"][0]["page_content"].lower()


def test_influenza_sample_uses_correct_chunk(rag_service):
    result = run_sample_case(rag_service, SAMPLE_QUERY_CASES[1])

    assert result["sources"][0]["metadata"]["source"] == "influenza_factsheet.pdf"
    assert "influenza" in result["answer"].lower() or "fever" in result["answer"].lower()
