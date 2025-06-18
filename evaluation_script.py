#!/usr/bin/env python3
"""
Evaluation Script for Code QA System
Measures the quality of the MCP server's answers against reference Q&A pairs.
"""

import json
import os
import sys
from typing import List
import logging
from dataclasses import dataclass
import requests
import time

# NLP evaluation imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rouge_score import rouge_scorer
import nltk

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Import our RAG system
from rag.mcp_code_qa_server import RAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """Represents a question-answer pair for evaluation"""

    question: str
    reference_answer: str
    category: str = "general"


@dataclass
class EvaluationResult:
    """Results of evaluating a single Q&A pair"""

    question: str
    reference_answer: str
    generated_answer: str
    semantic_similarity: float
    rouge_l: float
    overall_score: float
    category: str = "general"


class QAEvaluator:
    """Evaluates Q&A system performance using multiple metrics"""

    def __init__(self):
        self.sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def evaluate_single(
        self, question: str, reference: str, generated: str, category: str = "general"
    ) -> EvaluationResult:
        """Evaluate a single Q&A pair"""

        # Semantic similarity using sentence embeddings
        ref_embedding = self.sentence_encoder.encode([reference])
        gen_embedding = self.sentence_encoder.encode([generated])
        semantic_sim = float(cosine_similarity(ref_embedding, gen_embedding)[0][0])

        # ROUGE-L score
        rouge_scores = self.rouge_scorer.score(reference, generated)
        rouge_l = float(rouge_scores["rougeL"].fmeasure)

        # Overall score (weighted combination) - semantic similarity and ROUGE-L only
        overall_score = 0.7 * semantic_sim + 0.3 * rouge_l

        return EvaluationResult(
            question=question,
            reference_answer=reference,
            generated_answer=generated,
            semantic_similarity=semantic_sim,
            rouge_l=rouge_l,
            overall_score=overall_score,
            category=category,
        )

    def evaluate_batch(
        self, qa_pairs: List[QAPair], rag_system: RAGSystem
    ) -> List[EvaluationResult]:
        """Evaluate multiple Q&A pairs"""
        results = []

        for qa_pair in qa_pairs:
            logger.info(f"Evaluating: {qa_pair.question[:50]}...")

            # Generate answer using RAG system
            generated_answer = rag_system.answer_question(qa_pair.question)

            # Evaluate the answer
            result = self.evaluate_single(
                qa_pair.question,
                qa_pair.reference_answer,
                generated_answer,
                qa_pair.category,
            )

            results.append(result)

        return results


class GripQALoader:
    """Loads the Grip QA dataset for evaluation"""

    @staticmethod
    def load_from_github(
        repo_url: str = "https://api.github.com/repos/Modelcode-ai/grip_qa/contents",
    ) -> List[QAPair]:
        """Load Q&A pairs from the Grip QA GitHub repository"""
        qa_pairs = []

        try:
            # Get repository contents
            response = requests.get(repo_url)
            response.raise_for_status()
            contents = response.json()

            # Group files by their number (e.g., 0001.q.md and 0001.a.md)
            qa_files = {}

            for item in contents:
                if item["name"].endswith(".q.md") or item["name"].endswith(".a.md"):
                    # Extract the number from filename (e.g., "0001" from "0001.q.md")
                    file_number = item["name"].split(".")[0]

                    if file_number not in qa_files:
                        qa_files[file_number] = {}

                    if item["name"].endswith(".q.md"):
                        qa_files[file_number]["question_url"] = item["download_url"]
                    elif item["name"].endswith(".a.md"):
                        qa_files[file_number]["answer_url"] = item["download_url"]

            # Process each Q&A pair
            for file_number, urls in qa_files.items():
                if "question_url" in urls and "answer_url" in urls:
                    try:
                        # Download question file
                        question_response = requests.get(urls["question_url"])
                        question_response.raise_for_status()
                        question_text = question_response.text.strip()

                        # Download answer file
                        answer_response = requests.get(urls["answer_url"])
                        answer_response.raise_for_status()
                        answer_text = answer_response.text.strip()

                        # Create QAPair
                        qa_pairs.append(
                            QAPair(
                                question=question_text,
                                reference_answer=answer_text,
                                category="general",  # You might want to infer category from content
                            )
                        )

                    except Exception as e:
                        logger.warning(f"Error processing Q&A pair {file_number}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error loading from GitHub: {e}")
            # Fallback to local sample data
            return GripQALoader.load_sample_data()

        return qa_pairs

    @staticmethod
    def load_sample_data() -> List[QAPair]:
        """Load sample Q&A pairs for testing"""
        return [
            QAPair(
                question="What does the Server class do?",
                reference_answer="The Server class is the main MCP server implementation that handles tool registration and execution for code Q&A functionality.",
                category="class",
            ),
            QAPair(
                question="How is the RAG system implemented?",
                reference_answer="The RAG system is implemented using a combination of code parsing, vector storage with ChromaDB, and semantic search to retrieve relevant code chunks for answering questions.",
                category="architecture",
            ),
            QAPair(
                question="How does the CodeParser parse Python files?",
                reference_answer="The CodeParser uses Python's AST module to parse files into logical chunks like classes, functions, and methods, extracting metadata and dependencies for each chunk.",
                category="method",
            ),
            QAPair(
                question="What is the purpose of the VectorStore class?",
                reference_answer="The VectorStore class handles vector storage and retrieval using ChromaDB, encoding code chunks as vectors for semantic search capabilities.",
                category="class",
            ),
            QAPair(
                question="How are code chunks created?",
                reference_answer="Code chunks are created by parsing Python AST nodes to extract logical code blocks like functions, classes, and methods, along with their metadata and dependencies.",
                category="process",
            ),
        ]


class EvaluationReporter:
    """Generates evaluation reports"""

    @staticmethod
    def generate_report(
        results: List[EvaluationResult], output_file: str = "evaluation_report.json"
    ):
        """Generate a comprehensive evaluation report"""

        # Calculate aggregate metrics
        total_results = len(results)
        avg_semantic_sim = np.mean([r.semantic_similarity for r in results])
        avg_rouge_l = np.mean([r.rouge_l for r in results])
        avg_overall = np.mean([r.overall_score for r in results])

        # Category breakdown
        categories = {}
        for result in results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

        category_stats = {}
        for category, cat_results in categories.items():
            category_stats[category] = {
                "count": len(cat_results),
                "avg_semantic_similarity": np.mean(
                    [r.semantic_similarity for r in cat_results]
                ),
                "avg_rouge_l": np.mean([r.rouge_l for r in cat_results]),
                "avg_overall": np.mean([r.overall_score for r in cat_results]),
            }

        # Create report
        report = {
            "evaluation_summary": {
                "total_questions": total_results,
                "average_semantic_similarity": float(avg_semantic_sim),
                "average_rouge_l": float(avg_rouge_l),
                "average_overall_score": float(avg_overall),
            },
            "category_breakdown": category_stats,
            "detailed_results": [
                {
                    "question": r.question,
                    "reference_answer": r.reference_answer,
                    "generated_answer": r.generated_answer,
                    "semantic_similarity": float(r.semantic_similarity),
                    "rouge_l": float(r.rouge_l),
                    "overall_score": float(r.overall_score),
                    "category": r.category,
                }
                for r in results
            ],
        }

        # Save report
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        return report

    @staticmethod
    def print_summary(results: List[EvaluationResult]):
        """Print a summary of evaluation results"""
        total = len(results)
        avg_overall = np.mean([r.overall_score for r in results])
        avg_semantic = np.mean([r.semantic_similarity for r in results])
        avg_rouge = np.mean([r.rouge_l for r in results])

        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total Questions Evaluated: {total}")
        print(f"Average Overall Score: {avg_overall:.3f}")
        print(f"Average Semantic Similarity: {avg_semantic:.3f}")
        print(f"Average ROUGE-L: {avg_rouge:.3f}")

        # Score interpretation
        if avg_overall >= 0.8:
            print("🟢 Excellent performance!")
        elif avg_overall >= 0.6:
            print("🟡 Good performance")
        elif avg_overall >= 0.4:
            print("🟠 Fair performance")
        else:
            print("🔴 Needs improvement")

        print("\nTop 3 Best Performing Questions:")
        sorted_results = sorted(results, key=lambda x: x.overall_score, reverse=True)
        for i, result in enumerate(sorted_results[:3]):
            print(
                f"{i+1}. Score: {result.overall_score:.3f} - {result.question[:60]}..."
            )

        print("\nTop 3 Worst Performing Questions:")
        for i, result in enumerate(sorted_results[-3:]):
            print(
                f"{i+1}. Score: {result.overall_score:.3f} - {result.question[:60]}..."
            )


def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Code QA System")
    parser.add_argument(
        "--repo-path", required=True, help="Path to the repository to evaluate on"
    )
    parser.add_argument(
        "--qa-source",
        default="github",
        choices=["github", "sample"],
        help="Source of Q&A pairs (github or sample)",
    )
    parser.add_argument(
        "--output",
        default="evaluation_report.json",
        help="Output file for evaluation report",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check if repository exists
    if not os.path.exists(args.repo_path):
        print(f"Error: Repository path '{args.repo_path}' does not exist.")
        sys.exit(1)

    print("Starting Code QA System Evaluation...")
    print(f"Repository: {args.repo_path}")
    print(f"Q&A Source: {args.qa_source}")

    # Initialize RAG system and index repository
    print("\n1. Initializing RAG System...")
    rag_system = RAGSystem(args.repo_path)

    print("2. Indexing repository...")
    start_time = time.time()
    rag_system.index_repository()
    index_time = time.time() - start_time
    print(f"   Indexing completed in {index_time:.2f} seconds")
    print(f"   Indexed {len(rag_system.chunks)} code chunks")

    # Load Q&A pairs
    print("\n3. Loading Q&A pairs...")
    if args.qa_source == "github":
        qa_pairs = GripQALoader.load_from_github()
    else:
        qa_pairs = GripQALoader.load_sample_data()

    print(f"   Loaded {len(qa_pairs)} Q&A pairs")

    if not qa_pairs:
        print("Error: No Q&A pairs loaded. Exiting.")
        sys.exit(1)

    # Initialize evaluator
    print("\n4. Initializing evaluator...")
    evaluator = QAEvaluator()

    # Run evaluation
    print("\n5. Running evaluation...")
    start_time = time.time()
    results = evaluator.evaluate_batch(qa_pairs, rag_system)
    eval_time = time.time() - start_time
    print(f"   Evaluation completed in {eval_time:.2f} seconds")

    # Generate and save report
    print("\n6. Generating report...")
    report = EvaluationReporter.generate_report(results, args.output)
    print(f"   Report saved to: {args.output}")

    # Print summary
    EvaluationReporter.print_summary(results)

    # Additional analysis
    print(f"\n" + "=" * 50)
    print("DETAILED ANALYSIS")
    print("=" * 50)

    # Performance by category
    categories = {}
    for result in results:
        if result.category not in categories:
            categories[result.category] = []
        categories[result.category].append(result)

    print("\nPerformance by Category:")
    for category, cat_results in categories.items():
        avg_score = np.mean([r.overall_score for r in cat_results])
        print(f"  {category}: {avg_score:.3f} ({len(cat_results)} questions)")

    # Performance distribution
    scores = [r.overall_score for r in results]
    excellent = len([s for s in scores if s >= 0.8])
    good = len([s for s in scores if 0.6 <= s < 0.8])
    fair = len([s for s in scores if 0.4 <= s < 0.6])
    poor = len([s for s in scores if s < 0.4])

    print(f"\nScore Distribution:")
    print(f"  Excellent (≥0.8): {excellent} ({excellent/len(scores)*100:.1f}%)")
    print(f"  Good (0.6-0.8): {good} ({good/len(scores)*100:.1f}%)")
    print(f"  Fair (0.4-0.6): {fair} ({fair/len(scores)*100:.1f}%)")
    print(f"  Poor (<0.4): {poor} ({poor/len(scores)*100:.1f}%)")

    print(f"\nSystem Performance Metrics:")
    print(f"  Indexing time: {index_time:.2f} seconds")
    print(f"  Average query time: {eval_time/len(qa_pairs):.2f} seconds")
    print(f"  Total chunks indexed: {len(rag_system.chunks)}")

    return report


if __name__ == "__main__":
    main()
