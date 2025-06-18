#!/usr/bin/env python3
"""
Main Evaluation Script (eval.py)
Unified RAG evaluation using NLP and LLM judge methods.
"""

import os
import sys
import time
import statistics
import argparse
import logging

from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from .qa_loader import QADataLoader, QAPair
from .evaluators.nlp_evaluator import NLPEvaluator, NLPScores
from .evaluators.llm_judge_evaluator import LLMJudgeEvaluator, JudgeScores
from .evaluation_reporter import EvaluationReporter

# Add the parent directory to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag_system import RAGSystem

load_dotenv()
logger = logging.getLogger(__name__)

# Constants
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


@dataclass
class EvaluationResult:
    """Unified container for evaluation results"""

    question: str
    reference_answer: str
    generated_answer: str
    nlp_scores: Optional[NLPScores] = None
    judge_scores: Optional[JudgeScores] = None
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "question": self.question,
            "reference_answer": self.reference_answer,
            "generated_answer": self.generated_answer,
            "execution_time": self.execution_time,
        }

        if self.nlp_scores:
            result["nlp_scores"] = asdict(self.nlp_scores)

        if self.judge_scores:
            result["judge_scores"] = dict(self.judge_scores)

        return result


class RAGEvaluationSuite:
    """Main evaluation suite that coordinates different evaluation methods"""

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.rag_system = RAGSystem(repo_path)
        self.nlp_evaluator = None
        self.llm_judge_evaluator = None

        # Initialize RAG system with clean progress indicator
        print("\n1. Initializing RAG System...")
        start_time = time.time()
        print("2. Indexing repository...")
        self.rag_system.index_repository()
        index_time = time.time() - start_time
        chunks_count = len(getattr(self.rag_system, "chunks", []))
        print(f"   Indexing completed in {index_time:.2f} seconds")
        print(f"   Indexed {chunks_count} code chunks")

    def _init_nlp_evaluator(self):
        """Initialize NLP evaluator if not already done"""
        if self.nlp_evaluator is None:
            print("3. Initializing evaluator...")
            self.nlp_evaluator = NLPEvaluator()

    def _init_llm_judge_evaluator(self):
        """Initialize LLM judge evaluator if not already done"""
        if self.llm_judge_evaluator is None:
            if not hasattr(self, "_judge_init_printed"):
                print("   Initializing LLM judge...")
                self._judge_init_printed = True
            self.llm_judge_evaluator = LLMJudgeEvaluator(ANTHROPIC_API_KEY or "")

    def evaluate_batch(
        self, qa_pairs: List[QAPair], methods: List[Literal["nlp", "judge"]]
    ) -> List[EvaluationResult]:
        """Evaluate a batch of Q&A pairs using specified methods"""
        results = []

        # Initialize required evaluators
        if "nlp" in methods:
            self._init_nlp_evaluator()
        if "judge" in methods:
            self._init_llm_judge_evaluator()

        total_pairs = len(qa_pairs)
        print(f"\n4. Running evaluation...")
        print(f"   Evaluating {total_pairs} questions using {', '.join(methods)}...")

        for i, qa_pair in enumerate(qa_pairs, 1):
            # Clean progress indicator like the old script
            logger.info(f"Evaluating [{i}/{total_pairs}]: {qa_pair.question[:50]}...")

            start_time = time.time()

            # Generate answer using RAG system
            generated_answer = self.rag_system.answer_question(qa_pair.question)

            # Initialize result
            result = EvaluationResult(
                question=qa_pair.question,
                reference_answer=qa_pair.reference_answer,
                generated_answer=generated_answer,
            )

            # Run NLP evaluation if requested
            if "nlp" in methods and self.nlp_evaluator:
                try:
                    nlp_scores = self.nlp_evaluator.evaluate_answer(
                        qa_pair.reference_answer, generated_answer
                    )
                    if nlp_scores:  # Only assign if not None
                        result.nlp_scores = nlp_scores
                        logger.info(
                            f"      NLP scores: Overall={nlp_scores.overall_score:.3f}, "
                            f"Semantic={nlp_scores.semantic_similarity:.3f}, "
                            f"ROUGE-L={nlp_scores.rouge_l:.3f}"
                        )
                except Exception as e:
                    logger.warning(f"NLP evaluation failed for question {i}: {e}")

            # Run LLM judge evaluation if requested
            if "judge" in methods and self.llm_judge_evaluator:
                try:
                    judge_scores = self.llm_judge_evaluator.evaluate_answer(
                        qa_pair.question, qa_pair.reference_answer, generated_answer
                    )
                    if judge_scores:  # Only assign if not None
                        result.judge_scores = judge_scores
                        logger.info(
                            f"      Judge scores: Correctness={judge_scores['correctness']}, "
                            f"Completeness={judge_scores['completeness']}, "
                            f"Clarity={judge_scores['clarity']}"
                        )
                except Exception as e:
                    logger.warning(f"LLM judge evaluation failed for question {i}: {e}")

            result.execution_time = time.time() - start_time
            results.append(result)

        return results


def print_simple_summary(results: List[EvaluationResult], methods: List[str]):
    """Print a simple, clean summary to console"""
    total = len(results)
    avg_time = statistics.mean([r.execution_time for r in results]) if results else 0

    print(f"\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Questions Evaluated: {total}")
    print(f"Average time per question: {avg_time:.1f}s")

    # NLP Summary
    if "nlp" in methods and any(r.nlp_scores for r in results):
        nlp_scores = [r.nlp_scores.overall_score for r in results if r.nlp_scores]
        avg_nlp = statistics.mean(nlp_scores)
        print(f"Average NLP Score: {avg_nlp:.3f}")

        # Simple performance indicator
        if avg_nlp >= 0.8:
            print("🟢 Excellent performance!")
        elif avg_nlp >= 0.6:
            print("🟡 Good performance")
        elif avg_nlp >= 0.4:
            print("🟠 Fair performance")
        else:
            print("🔴 Needs improvement")

    # Judge Summary
    if "judge" in methods and any(r.judge_scores for r in results):
        judge_results = [r for r in results if r.judge_scores]
        avg_scores = {
            metric: statistics.mean(
                [
                    r.judge_scores[metric]
                    for r in judge_results
                    if r.judge_scores is not None
                ]
            )
            for metric in ["correctness", "completeness", "clarity"]
        }
        overall_judge = statistics.mean(list(avg_scores.values()))

        print(f"Average LLM Judge Score: {overall_judge:.1f}/10")
        print(f"  Correctness: {avg_scores['correctness']:.1f}/10")
        print(f"  Completeness: {avg_scores['completeness']:.1f}/10")
        print(f"  Clarity: {avg_scores['clarity']:.1f}/10")

        # Simple performance indicator
        if overall_judge >= 8:
            print("🟢 Excellent performance!")
        elif overall_judge >= 6:
            print("🟡 Good performance")
        elif overall_judge >= 4:
            print("🟠 Fair performance")
        else:
            print("🔴 Needs improvement")

    print("=" * 50)


def main():
    """Main CLI interface for the evaluation suite"""
    parser = argparse.ArgumentParser(
        description="Unified RAG Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval.py --repo-path ./my_repo --methods nlp judge
  python eval.py --repo-path ./my_repo --methods nlp --qa-source sample
        """,
    )

    parser.add_argument(
        "--repo-path", required=True, help="Path to the repository to evaluate"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["nlp", "judge"],
        default=["nlp", "judge"],
        help="Evaluation methods to run (default: both)",
    )
    parser.add_argument(
        "--qa-source",
        choices=["github", "sample"],
        default="github",
        help="Source of Q&A pairs (default: github)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.repo_path):
        print(f"Error: Repository path '{args.repo_path}' does not exist.")
        sys.exit(1)

    if "judge" in args.methods and not ANTHROPIC_API_KEY:
        print(
            "Error: ANTHROPIC_API_KEY environment variable is required for LLM judge evaluation"
        )
        sys.exit(1)

    print("Starting RAG Evaluation Suite...")
    print(f"Repository: {args.repo_path}")
    print(f"Methods: {', '.join(args.methods)}")

    try:
        # Initialize evaluation suite
        suite = RAGEvaluationSuite(args.repo_path)

        # Load Q&A pairs
        print(f"\nLoading Q&A data from {args.qa_source}...")
        qa_pairs = []

        try:
            if args.qa_source == "github":
                qa_pairs = QADataLoader.load_from_github()
            elif args.qa_source == "sample":
                qa_pairs = QADataLoader.load_sample_data()
        except Exception as e:
            print(f"Error loading Q&A data: {e}")
            sys.exit(1)

        if not qa_pairs:
            print("Error: No Q&A pairs loaded. Exiting.")
            sys.exit(1)

        print(f"   Loaded {len(qa_pairs)} Q&A pairs")

        # Run evaluation
        start_time = time.time()
        results = suite.evaluate_batch(qa_pairs, args.methods)
        total_time = time.time() - start_time

        print(f"   Evaluation completed in {total_time:.2f} seconds")

        # Generate and save detailed report
        evaluation_config = {
            "repo_path": args.repo_path,
            "qa_source": args.qa_source,
            "methods": args.methods,
        }

        print("\n5. Generating report...")
        report_path = "evaluation/evaluation_report.json"
        report = EvaluationReporter.generate_report(
            results, report_path, evaluation_config
        )

        # Print simple summary to console
        print_simple_summary(results, args.methods)

        print(f"\nReport saved to: {report_path}")
        return report

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
