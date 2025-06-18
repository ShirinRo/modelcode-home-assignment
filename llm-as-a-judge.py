import os
import json
import statistics
import time
from typing import Dict, List, Optional, Union, Literal, TypedDict
from dataclasses import dataclass

from evaluation_script import GripQALoader  # assumes these classes are available
from rag.rag_system import RAGSystem  # (use your provided .py files)
import anthropic
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Set your Anthropic API key as an environment variable or directly here
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

### --- Type Definitions ---


class JudgeScores(TypedDict):
    """Type definition for judge evaluation scores"""

    correctness: int
    completeness: int
    clarity: int


@dataclass
class EvaluationResult:
    """Container for a single evaluation result"""

    question: str
    reference_answer: str
    generated_answer: str
    scores: JudgeScores

    def to_dict(self) -> Dict[str, Union[str, int]]:
        """Convert to dictionary for JSON serialization"""
        return {
            "question": self.question,
            "reference_answer": self.reference_answer,
            "generated_answer": self.generated_answer,
            "correctness": self.scores["correctness"],
            "completeness": self.scores["completeness"],
            "clarity": self.scores["clarity"],
        }


@dataclass
class EvaluationStats:
    """Statistics for a single evaluation criterion"""

    criterion: str
    average: float
    median: float
    minimum: int
    maximum: int

    def __str__(self) -> str:
        return (
            f"{self.criterion.capitalize()} stats:\n"
            f"  Average: {self.average:.2f}\n"
            f"  Median: {self.median:.2f}\n"
            f"  Min: {self.minimum}\n"
            f"  Max: {self.maximum}"
        )


ScoreType = Literal["correctness", "completeness", "clarity"]

### --- Function to Get Judge Score from Claude using Tool Calling ---


def validate_judge_scores(
    scores: Union[Dict[str, Union[int, str]], object],
) -> JudgeScores:
    """
    Validate and convert judge scores to proper type

    Args:
        scores: Raw scores dictionary from API response

    Returns:
        Validated JudgeScores object

    Raises:
        ValueError: If scores are invalid
    """
    # First check if it's a dictionary
    if not isinstance(scores, dict):
        raise ValueError(f"Expected dict, got {type(scores)}")

    required_keys: List[ScoreType] = ["correctness", "completeness", "clarity"]

    # Check all required keys are present
    for key in required_keys:
        if key not in scores:
            raise ValueError(f"Missing required key: {key}")

    # Validate and convert scores
    validated_scores: Dict[ScoreType, int] = {}
    for key in required_keys:
        value = scores[key]

        # Convert to int if it's a string
        if isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                raise ValueError(
                    f"Invalid score for {key}: {value} (cannot convert to int)"
                )

        if not isinstance(value, int):
            raise ValueError(
                f"Invalid score type for {key}: {type(value)} (expected int)"
            )

        if not (1 <= value <= 10):
            raise ValueError(f"Invalid score range for {key}: {value} (must be 1-10)")

        validated_scores[key] = value

    return JudgeScores(
        correctness=validated_scores["correctness"],
        completeness=validated_scores["completeness"],
        clarity=validated_scores["clarity"],
    )


def get_default_scores() -> JudgeScores:
    """Return default scores when evaluation fails"""
    return JudgeScores(correctness=5, completeness=5, clarity=5)


def ask_claude_judge(question: str, reference: str, generated: str) -> JudgeScores:
    """
    Uses Anthropic's tool calling to ensure structured JSON output

    Args:
        question: The original question
        reference: Reference answer
        generated: Generated answer to evaluate

    Returns:
        JudgeScores with correctness, completeness, and clarity scores (1-10)
    """
    prompt = f"""You are an expert code reviewer evaluating answers.

Question:
{question.strip()}

Reference Answer:
{reference.strip()}

Model's Answer:
{generated.strip()}

Evaluate the model's answer compared to the reference answer using the evaluate_answer tool. Consider:
- Correctness: Is the information factually accurate?
- Completeness: Does it address all key points from the reference?
- Clarity: Is it well-structured and easy to understand?

Use the `evaluate_answer` tool to provide your evaluation scores."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=256,
                temperature=0,
                system="You are an impartial and precise code answer evaluator. Use the provided tool to score answers.",
                tools=[
                    {
                        "name": "evaluate_answer",
                        "description": "Evaluate an answer for correctness, completeness, and clarity",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "correctness": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 10,
                                    "description": "Score for factual correctness (1-10). How accurate is the information?",
                                },
                                "completeness": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 10,
                                    "description": "Score for completeness (1-10). Does it cover all important aspects?",
                                },
                                "clarity": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 10,
                                    "description": "Score for clarity (1-10). Is it clear and easy to understand?",
                                },
                            },
                            "required": ["correctness", "completeness", "clarity"],
                        },
                    }
                ],
                tool_choice={"type": "tool", "name": "evaluate_answer"},
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract the tool call result
            for block in response.content:
                if (
                    hasattr(block, "type")
                    and block.type == "tool_use"
                    and block.name == "evaluate_answer"
                ):
                    # Type check and validate the scores
                    raw_input = block.input
                    if not isinstance(raw_input, dict):
                        raise ValueError(
                            f"Expected dict from tool call, got {type(raw_input)}"
                        )

                    return validate_judge_scores(raw_input)

            # If we get here, no tool call was found
            raise ValueError("No tool call found in response")

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print("All attempts failed, returning default scores")
                return get_default_scores()
            time.sleep(2)

    # This should never be reached, but type checker needs it
    return get_default_scores()


### --- Load Questions from Github/Sample ---


def load_questions(source: Literal["github", "sample"] = "github"):
    """Load Q&A pairs from specified source"""
    if source == "github":
        qa_pairs = GripQALoader.load_from_github()
    else:
        qa_pairs = GripQALoader.load_sample_data()
    return qa_pairs


### --- Statistics Calculation ---


def calculate_stats(results: List[EvaluationResult]) -> List[EvaluationStats]:
    """
    Calculate statistics for all evaluation criteria

    Args:
        results: List of evaluation results

    Returns:
        List of EvaluationStats for each criterion
    """
    criteria: List[ScoreType] = ["correctness", "completeness", "clarity"]
    stats: List[EvaluationStats] = []

    for criterion in criteria:
        values = [result.scores[criterion] for result in results]

        if not values:  # Empty list check
            stats.append(
                EvaluationStats(
                    criterion=criterion, average=0.0, median=0.0, minimum=0, maximum=0
                )
            )
            continue

        stats.append(
            EvaluationStats(
                criterion=criterion,
                average=statistics.mean(values),
                median=statistics.median(values),
                minimum=min(values),
                maximum=max(values),
            )
        )

    return stats


### --- MAIN EVAL SCRIPT ---


def main(
    repo_path: str,
    qa_source: Literal["github", "sample"] = "github",
    output_file: str = "llm_judge_eval_results.json",
) -> None:
    """
    Main evaluation script

    Args:
        repo_path: Path to the repository to evaluate
        qa_source: Source of Q&A pairs ("github" or "sample")
        output_file: Output file for results
    """
    # Init RAG and index repo
    rag = RAGSystem(repo_path)
    print("Indexing repository...")
    rag.index_repository()

    print("Loading questions...")
    qa_pairs = load_questions(qa_source)
    print(f"Loaded {len(qa_pairs)} Q&A pairs")

    results: List[EvaluationResult] = []

    for i, qa in enumerate(qa_pairs, 1):
        print(f"\nEvaluating [{i}/{len(qa_pairs)}]: {qa.question[:60]}...")

        # Get generated answer from RAG
        generated = rag.answer_question(qa.question)

        # Get LLM judge scores using tool calling
        scores = ask_claude_judge(qa.question, qa.reference_answer, generated)
        print(f"  Scores: {scores}")

        # Create evaluation result
        result = EvaluationResult(
            question=qa.question,
            reference_answer=qa.reference_answer,
            generated_answer=generated,
            scores=scores,
        )
        results.append(result)

    # Save results
    results_dicts = [result.to_dict() for result in results]
    with open(output_file, "w") as f:
        json.dump(results_dicts, f, indent=2)

    print("\n======= LLM Judge Statistical Summary =======")
    stats = calculate_stats(results)
    for stat in stats:
        print(f"\n{stat}")


if __name__ == "__main__":
    main(repo_path="grip-no-tests", qa_source="github")
