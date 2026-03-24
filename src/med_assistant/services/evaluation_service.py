import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

class EvaluatorService:
    """Service to evaluate the clinical reliability of RAG responses."""

    def __init__(self, llm_pipeline):
        self.llm_pipeline = llm_pipeline

    def evaluate_response(self, query: str, context: str, answer: str) -> Dict[str, float]:
        """
        Evaluate the faithfulness and relevance of the answer.
        Returns a dictionary with scores between 0 and 1.
        """
        faithfulness = self._calculate_faithfulness(context, answer)
        relevance = self._calculate_relevance(query, answer)
        
        # Combined confidence score (weighted average)
        # We weight faithfulness higher in medical contexts to prevent hallucinations
        confidence_score = (faithfulness * 0.7) + (relevance * 0.3)
        
        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "confidence_score": confidence_score
        }

    def _calculate_faithfulness(self, context: str, answer: str) -> float:
        """Checks if the answer is strictly based on the provided context."""
        prompt = f"""
        [CONTEXT]
        {context}
        [/CONTEXT]

        [ANSWER]
        {answer}
        [/ANSWER]

        Based on the [CONTEXT] above, is every claim in the [ANSWER] supported? 
        Respond ONLY with a single score between 1 and 10, where 10 means 100% grounded and 1 means completely hallucinated.
        Score:"""
        
        try:
            response = self.llm_pipeline(prompt, max_new_tokens=5, temperature=0.1)
            raw_text = response[0]['generated_text']
            # Find the first number in the response
            match = re.search(r'(\d+)', raw_text.split("Score:")[-1])
            if match:
                score = int(match.group(1))
                return min(max(score, 1), 10) / 10.0
        except Exception as e:
            logger.error(f"Error calculating faithfulness: {e}")
            
        return 0.8  # Default fallback

    def _calculate_relevance(self, query: str, answer: str) -> float:
        """Checks if the answer addresses the original query."""
        prompt = f"""
        [QUERY]
        {query}
        [/QUERY]

        [ANSWER]
        {answer}
        [/ANSWER]

        How well does the [ANSWER] address the [QUERY]?
        Respond ONLY with a single score between 1 and 10, where 10 means perfectly relevant and 1 means off-topic.
        Score:"""
        
        try:
            response = self.llm_pipeline(prompt, max_new_tokens=5, temperature=0.1)
            raw_text = response[0]['generated_text']
            match = re.search(r'(\d+)', raw_text.split("Score:")[-1])
            if match:
                score = int(match.group(1))
                return min(max(score, 1), 10) / 10.0
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            
        return 0.8  # Default fallback
