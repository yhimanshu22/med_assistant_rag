"""Tests for Ragas score extraction (no model load required)."""

import math

import pandas as pd
import pytest

from med_assistant.services.evaluation_service import _extract_metric_score


class _FakeResult:
    def __init__(self, scores: dict):
        self._scores = scores
        self._repr_dict = scores

    def __getitem__(self, key: str):
        return self._scores[key]


def test_extract_metric_score_reads_valid_values():
    result = _FakeResult({"faithfulness": [0.55], "answer_relevancy": [0.72]})
    assert _extract_metric_score(result, "faithfulness") == pytest.approx(0.55)
    assert _extract_metric_score(result, "answer_relevancy") == pytest.approx(0.72)


def test_extract_metric_score_returns_zero_for_nan_not_eighty_percent():
    result = _FakeResult({"faithfulness": [math.nan], "answer_relevancy": [math.nan]})
    assert _extract_metric_score(result, "faithfulness") == 0.0
    assert _extract_metric_score(result, "answer_relevancy") == 0.0


def test_extract_metric_score_returns_zero_for_missing_metric():
    result = _FakeResult({"faithfulness": [0.4]})
    assert _extract_metric_score(result, "answer_relevancy") == 0.0


def test_extract_metric_score_handles_series_input():
    result = _FakeResult({"faithfulness": pd.Series([0.61])})
    assert _extract_metric_score(result, "faithfulness") == pytest.approx(0.61)
