"""
Fast demo runner — validates sample medical Q&A without loading Mistral-7B or ChromaDB.

Usage:
    uv run python -m tests.demo_sample_queries
"""

from __future__ import annotations

import textwrap

from tests.fixtures.medical_samples import SAMPLE_QUERY_CASES, WEAK_RETRIEVAL_CASE
from tests.fixtures.rag_test_utils import make_unit_rag_service, patch_retrieval, run_sample_case

DIVIDER = "=" * 72


def _wrap(text: str, width: int = 68) -> str:
    return textwrap.fill(text, width=width)


def _print_case(title: str, question: str, context: str, answer: str, sources: list) -> None:
    print(DIVIDER)
    print(title)
    print(DIVIDER)
    print(f"Question:\n  {_wrap(question)}\n")
    print(f"Retrieved context:\n  {_wrap(context)}\n")
    print(f"Answer:\n  {_wrap(answer)}\n")
    if sources:
        src = sources[0].get("metadata", {})
        print(f"Source: {src.get('source', 'unknown')} (page {src.get('page_number', '?')})")
    print()


def run_demo() -> None:
    print("\nMedAssist RAG — Fast Sample Query Demo (mocked LLM, no GPU)\n")

    service = make_unit_rag_service()

    passed = 0
    for i, case in enumerate(SAMPLE_QUERY_CASES, start=1):
        result = run_sample_case(service, case)
        answer_lower = result["answer"].lower()
        terms_ok = all(term.lower() in answer_lower for term in case["expected_terms"])
        status = "PASS" if terms_ok else "FAIL"
        passed += int(terms_ok)

        _print_case(
            f"Case {i}: {case['id']} [{status}]",
            case["question"],
            case["primary_chunk"].page_content,
            result["answer"],
            result["sources"],
        )

    patch_retrieval(
        service,
        primary_chunk=WEAK_RETRIEVAL_CASE["primary_chunk"],
        dense_distance=WEAK_RETRIEVAL_CASE["dense_distance"],
    )
    weak = service.answer_question(WEAK_RETRIEVAL_CASE["question"])
    refused = "strong enough evidence" in weak["answer"].lower()
    passed += int(refused)

    _print_case(
        f"Case {len(SAMPLE_QUERY_CASES) + 1}: weak_retrieval [{'PASS' if refused else 'FAIL'}]",
        WEAK_RETRIEVAL_CASE["question"],
        WEAK_RETRIEVAL_CASE["primary_chunk"].page_content,
        weak["answer"],
        weak["sources"],
    )

    total = len(SAMPLE_QUERY_CASES) + 1
    print(DIVIDER)
    print(f"Demo complete: {passed}/{total} checks passed")
    print("Pipeline stages exercised: retrieval -> threshold -> prompt -> grounded answer")
    print(DIVIDER + "\n")


if __name__ == "__main__":
    run_demo()
