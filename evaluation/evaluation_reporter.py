"""
Evaluation Reporter Module
Handles statistics calculation and detailed report generation.
"""

import json
import statistics
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EvaluationStats:
    """Statistics for evaluation metrics"""

    metric_name: str
    average: float
    median: float
    minimum: float
    maximum: float
    std_dev: float


class EvaluationReporter:
    """Handles statistics calculation and comprehensive report generation"""

    @staticmethod
    def calculate_statistics(results: List[Any]) -> Dict[str, List[EvaluationStats]]:
        """Calculate statistics for all available metrics"""
        stats = {"nlp": [], "judge": []}

        # NLP statistics
        if any(r.nlp_scores for r in results):
            nlp_results = [r for r in results if r.nlp_scores]

            for metric in ["semantic_similarity", "rouge_l", "overall_score"]:
                values = [getattr(r.nlp_scores, metric) for r in nlp_results]
                if values:
                    stats["nlp"].append(
                        EvaluationStats(
                            metric_name=f"NLP {metric.replace('_', ' ').title()}",
                            average=statistics.mean(values),
                            median=statistics.median(values),
                            minimum=min(values),
                            maximum=max(values),
                            std_dev=(
                                statistics.stdev(values) if len(values) > 1 else 0.0
                            ),
                        )
                    )

        # Judge statistics
        if any(r.judge_scores for r in results):
            judge_results = [r for r in results if r.judge_scores]

            for metric in ["correctness", "completeness", "clarity"]:
                values = [r.judge_scores[metric] for r in judge_results]
                if values:
                    stats["judge"].append(
                        EvaluationStats(
                            metric_name=f"Judge {metric.title()}",
                            average=statistics.mean(values),
                            median=statistics.median(values),
                            minimum=min(values),
                            maximum=max(values),
                            std_dev=(
                                statistics.stdev(values) if len(values) > 1 else 0.0
                            ),
                        )
                    )

        return stats

    @staticmethod
    def _get_performance_distribution(
        results: List[Any], score_type: str
    ) -> Dict[str, Any]:
        """Calculate performance distribution for a given score type"""
        if score_type == "nlp":
            scores = [r.nlp_scores.overall_score for r in results if r.nlp_scores]
            thresholds = [
                (0.8, "excellent"),
                (0.6, "good"),
                (0.4, "fair"),
                (0.0, "poor"),
            ]
        elif score_type == "judge":
            scores = [
                statistics.mean(
                    [
                        r.judge_scores[m]
                        for m in ["correctness", "completeness", "clarity"]
                    ]
                )
                for r in results
                if r.judge_scores
            ]
            thresholds = [(8, "excellent"), (6, "good"), (4, "fair"), (0, "poor")]
        else:
            return {}

        if not scores:
            return {}

        distribution = {}
        for i, (threshold, label) in enumerate(thresholds):
            if i == 0:  # First threshold (highest)
                count = len([s for s in scores if s >= threshold])
            else:
                prev_threshold = thresholds[i - 1][0]
                count = len([s for s in scores if threshold <= s < prev_threshold])

            distribution[label] = {
                "count": count,
                "percentage": count / len(scores) * 100,
            }

        return distribution

    @staticmethod
    def _get_top_bottom_performers(
        results: List[Any], method: str, n: int = 3
    ) -> Dict[str, List[Dict]]:
        """Get top and bottom performing questions for a given method"""
        performers = {"top": [], "bottom": []}

        if method == "nlp" and any(r.nlp_scores for r in results):
            nlp_results = [r for r in results if r.nlp_scores]
            sorted_results = sorted(
                nlp_results, key=lambda x: x.nlp_scores.overall_score, reverse=True
            )

            for result in sorted_results[:n]:
                performers["top"].append(
                    {
                        "question": result.question,
                        "score": result.nlp_scores.overall_score,
                        "semantic_similarity": result.nlp_scores.semantic_similarity,
                        "rouge_l": result.nlp_scores.rouge_l,
                    }
                )

            for result in sorted_results[-n:]:
                performers["bottom"].append(
                    {
                        "question": result.question,
                        "score": result.nlp_scores.overall_score,
                        "semantic_similarity": result.nlp_scores.semantic_similarity,
                        "rouge_l": result.nlp_scores.rouge_l,
                    }
                )

        elif method == "judge" and any(r.judge_scores for r in results):
            judge_results = [r for r in results if r.judge_scores]
            sorted_results = sorted(
                judge_results,
                key=lambda x: statistics.mean(
                    [
                        x.judge_scores[m]
                        for m in ["correctness", "completeness", "clarity"]
                    ]
                ),
                reverse=True,
            )

            for result in sorted_results[:n]:
                avg_score = statistics.mean(
                    [
                        result.judge_scores[m]
                        for m in ["correctness", "completeness", "clarity"]
                    ]
                )
                performers["top"].append(
                    {
                        "question": result.question,
                        "average_score": avg_score,
                        "correctness": result.judge_scores["correctness"],
                        "completeness": result.judge_scores["completeness"],
                        "clarity": result.judge_scores["clarity"],
                    }
                )

            for result in sorted_results[-n:]:
                avg_score = statistics.mean(
                    [
                        result.judge_scores[m]
                        for m in ["correctness", "completeness", "clarity"]
                    ]
                )
                performers["bottom"].append(
                    {
                        "question": result.question,
                        "average_score": avg_score,
                        "correctness": result.judge_scores["correctness"],
                        "completeness": result.judge_scores["completeness"],
                        "clarity": result.judge_scores["clarity"],
                    }
                )

        return performers

    @staticmethod
    def _get_recommendations(results: List[Any], methods: List[str]) -> List[str]:
        """Generate performance recommendations based on evaluation results"""
        recommendations = []

        if "nlp" in methods and any(r.nlp_scores for r in results):
            nlp_scores = [r.nlp_scores.overall_score for r in results if r.nlp_scores]
            avg_nlp = statistics.mean(nlp_scores)

            if avg_nlp < 0.6:
                recommendations.extend(
                    [
                        "Consider improving code chunk extraction and parsing",
                        "Review semantic similarity thresholds",
                        "Enhance context window for answer generation",
                        "Consider fine-tuning embedding models on code-specific data",
                    ]
                )

        if "judge" in methods and any(r.judge_scores for r in results):
            judge_results = [r for r in results if r.judge_scores]
            avg_correctness = statistics.mean(
                [r.judge_scores["correctness"] for r in judge_results]
            )
            avg_completeness = statistics.mean(
                [r.judge_scores["completeness"] for r in judge_results]
            )
            avg_clarity = statistics.mean(
                [r.judge_scores["clarity"] for r in judge_results]
            )

            if avg_correctness < 6:
                recommendations.extend(
                    [
                        "Focus on improving factual accuracy of generated answers",
                        "Consider adding fact-checking mechanisms",
                    ]
                )
            if avg_completeness < 6:
                recommendations.extend(
                    [
                        "Improve context retrieval to capture more comprehensive information",
                        "Consider increasing the number of retrieved chunks",
                    ]
                )
            if avg_clarity < 6:
                recommendations.extend(
                    [
                        "Enhance answer formatting and structure",
                        "Improve natural language generation clarity",
                    ]
                )

        return recommendations

    @staticmethod
    def generate_report(
        results: List[Any], output_file: str, evaluation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""

        stats = EvaluationReporter.calculate_statistics(results)
        methods = evaluation_config.get("methods", [])

        # Create comprehensive report
        report = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "repository_path": evaluation_config.get("repo_path"),
                "qa_source": evaluation_config.get("qa_source"),
                "qa_file": evaluation_config.get("qa_file"),
                "evaluation_methods": methods,
                "total_questions": len(results),
                "avg_execution_time": (
                    statistics.mean([r.execution_time for r in results])
                    if results
                    else 0
                ),
            },
            "summary_statistics": {
                "nlp": {
                    stat.metric_name: {
                        "average": stat.average,
                        "median": stat.median,
                        "min": stat.minimum,
                        "max": stat.maximum,
                        "std_dev": stat.std_dev,
                    }
                    for stat in stats["nlp"]
                },
                "judge": {
                    stat.metric_name: {
                        "average": stat.average,
                        "median": stat.median,
                        "min": stat.minimum,
                        "max": stat.maximum,
                        "std_dev": stat.std_dev,
                    }
                    for stat in stats["judge"]
                },
            },
            "performance_analysis": {},
            "detailed_results": [result.to_dict() for result in results],
            "recommendations": EvaluationReporter._get_recommendations(
                results, methods
            ),
        }

        # Add performance distributions
        if "nlp" in methods:
            report["performance_analysis"]["nlp_distribution"] = (
                EvaluationReporter._get_performance_distribution(results, "nlp")
            )
            report["performance_analysis"]["nlp_performers"] = (
                EvaluationReporter._get_top_bottom_performers(results, "nlp")
            )

        if "judge" in methods:
            report["performance_analysis"]["judge_distribution"] = (
                EvaluationReporter._get_performance_distribution(results, "judge")
            )
            report["performance_analysis"]["judge_performers"] = (
                EvaluationReporter._get_top_bottom_performers(results, "judge")
            )

        # Save report
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report saved to: {output_file}")
        return report
