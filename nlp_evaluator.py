"""
NLP Evaluator Module
Handles NLP-based evaluation using statistical metrics.
"""

import logging
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


@dataclass
class NLPScores:
    """Container for NLP evaluation scores"""

    semantic_similarity: float
    rouge_l: float
    overall_score: float


class NLPEvaluator:
    """Handles NLP-based evaluation using statistical metrics"""

    def __init__(self):
        logger.info("Initializing NLP evaluator...")
        self.sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        logger.info("NLP evaluator initialized")

    def evaluate_answer(self, reference: str, generated: str) -> NLPScores:
        """Evaluate a single answer using NLP metrics"""
        # Semantic similarity using sentence embeddings
        ref_embedding = self.sentence_encoder.encode([reference])
        gen_embedding = self.sentence_encoder.encode([generated])
        semantic_sim = float(cosine_similarity(ref_embedding, gen_embedding)[0][0])

        # ROUGE-L score
        rouge_scores = self.rouge_scorer.score(reference, generated)
        rouge_l = float(rouge_scores["rougeL"].fmeasure)

        # Overall score (weighted combination)
        overall_score = 0.7 * semantic_sim + 0.3 * rouge_l

        return NLPScores(
            semantic_similarity=semantic_sim,
            rouge_l=rouge_l,
            overall_score=overall_score,
        )
