import logging
import math
from typing import Any, Dict
import pandas as pd
from datasets import Dataset

# Ragas imports
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

logger = logging.getLogger(__name__)


def _extract_metric_score(result: Any, key: str) -> float:
    """Read a Ragas metric score; return 0.0 (not a fake high score) when missing or NaN."""
    try:
        values = result[key]
        if isinstance(values, pd.Series):
            values = values.tolist()
        if not isinstance(values, (list, tuple)) or not values:
            logger.warning("Ragas metric %s returned no values", key)
            return 0.0

        numeric = []
        for value in values:
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isnan(score):
                numeric.append(score)

        if not numeric:
            logger.warning("Ragas metric %s returned only NaN/invalid values", key)
            return 0.0

        return sum(numeric) / len(numeric)
    except KeyError:
        available = sorted(getattr(result, "_repr_dict", {}).keys())
        logger.warning("Ragas result missing metric %s (available: %s)", key, available)
        return 0.0
    except (TypeError, ValueError, IndexError) as exc:
        logger.warning("Failed to parse Ragas metric %s: %s", key, exc)
        return 0.0

class EvaluatorService:
    """Service to evaluate the clinical reliability of RAG responses using Ragas."""

    def __init__(self, llm, embeddings):
        """
        Initialize with a LangChain LLM and embeddings.
        Compatible with local HuggingFacePipeline (and other LangChain LLMs).
        """
        self.llm = LangchainLLMWrapper(llm)
        self.embeddings = LangchainEmbeddingsWrapper(embeddings)
        
        # Configure metrics
        self.metrics = [faithfulness, answer_relevancy]
        for metric in self.metrics:
            metric.llm = self.llm
            # Ensure the internal LLM wrapper for the metric also respects n=1
            # Some Ragas versions might use different attribute names, so we set it where possible
            if hasattr(metric.llm, 'n'):
                metric.llm.n = 1
            if hasattr(metric.llm, 'llm') and hasattr(metric.llm.llm, 'n'):
                metric.llm.llm.n = 1
            if hasattr(metric, 'embeddings'):
                metric.embeddings = self.embeddings


    def evaluate_response(self, query: str, context: str, answer: str) -> Dict[str, float]:
        """
        Evaluate the faithfulness and relevance of the answer using Ragas.
        Returns a dictionary with scores between 0 and 1.
        """
        # Ragas 0.4+ expects these column names (legacy question/contexts/answer are ignored).
        data = {
            "user_input": [query],
            "retrieved_contexts": [[context]],
            "response": [answer],
        }
        dataset = Dataset.from_dict(data)

        try:
            # Perform evaluation
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings,
                run_config=RunConfig(max_workers=1)
            )
            
            f_score = _extract_metric_score(result, "faithfulness")
            r_score = _extract_metric_score(result, "answer_relevancy")

            
            # Combined confidence score (weighted average)
            confidence_score = (f_score * 0.7) + (r_score * 0.3)

            return {
                "faithfulness": float(f_score),
                "relevance": float(r_score),
                "confidence_score": float(confidence_score)
            }

        except Exception as e:
            logger.error(f"Error during Ragas evaluation: {e}")
            # Fallback to defaults
            return {
                "faithfulness": 0.2,
                "relevance": 0.2,
                "confidence_score": 0.2
            }
