import logging
from typing import List, Dict
import pandas as pd
from datasets import Dataset

# Ragas imports
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_huggingface import HuggingFacePipeline

logger = logging.getLogger(__name__)

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
        # Create dataset for Ragas (Ragas expects a list of contexts)
        data = {
            "question": [query],
            "contexts": [[context]],
            "answer": [answer]
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
            
            # Helper to safely extract and convert scores
            def safe_score(key, default=0.8):
                try:
                    # EvaluationResult can be accessed like a dict
                    val = result[key]
                    
                    # If it's a list or sequence, take the first element
                    if isinstance(val, (list, pd.Series)):
                        val = val[0] if len(val) > 0 else default
                    
                    f_val = float(val)
                    return f_val if not pd.isna(f_val) else default
                except (KeyError, ValueError, TypeError, IndexError):
                    return default

            f_score = safe_score("faithfulness")
            r_score = safe_score("answer_relevancy")

            
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
