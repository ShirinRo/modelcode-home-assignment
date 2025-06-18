"""
LLM Judge Evaluator Module
Handles LLM-as-a-judge evaluation using Anthropic's Claude.
"""

import time
import logging
from typing import Union, Dict, TypedDict
from anthropic import Anthropic

logger = logging.getLogger(__name__)


class JudgeScores(TypedDict):
    """Type definition for LLM judge evaluation scores"""

    correctness: int
    completeness: int
    clarity: int


class LLMJudgeEvaluator:
    """Handles LLM-as-a-judge evaluation using Anthropic's Claude"""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Anthropic API key is required for LLM judge evaluation")

        logger.info("Initializing LLM judge evaluator...")
        self.client = Anthropic(api_key=api_key)
        logger.info("LLM judge evaluator initialized")

    def _validate_scores(
        self, scores: Union[Dict[str, Union[int, str]], object]
    ) -> JudgeScores:
        """Validate and convert judge scores to proper type"""
        if not isinstance(scores, dict):
            raise ValueError(f"Expected dict, got {type(scores)}")

        required_keys = ["correctness", "completeness", "clarity"]
        validated_scores = {}

        for key in required_keys:
            if key not in scores:
                raise ValueError(f"Missing required key: {key}")

            value = scores[key]
            if isinstance(value, str):
                try:
                    value = int(value)
                except ValueError:
                    raise ValueError(f"Invalid score for {key}: {value}")

            if not isinstance(value, int) or not (1 <= value <= 10):
                raise ValueError(f"Invalid score for {key}: {value} (must be 1-10)")

            validated_scores[key] = value

        return JudgeScores(
            correctness=validated_scores["correctness"],
            completeness=validated_scores["completeness"],
            clarity=validated_scores["clarity"],
        )

    def evaluate_answer(
        self, question: str, reference: str, generated: str
    ) -> JudgeScores:
        """Evaluate answer using LLM judge with tool calling"""
        prompt = f"""You are an expert code reviewer evaluating answers.

Question:
{question.strip()}

Expected Answer:
{reference.strip()}

Model's Answer:
{generated.strip()}

Evaluate the model's answer compared to the expected answer using the evaluate_answer tool. 
Consider:
- Correctness: Is the information factually accurate?
- Completeness: Does it address all key points from the expected answer?
- Clarity: Is it well-structured and easy to understand?

Use the `evaluate_answer` tool to provide your evaluation scores."""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
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
                                        "description": "Score for factual correctness (1-10)",
                                    },
                                    "completeness": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 10,
                                        "description": "Score for completeness (1-10)",
                                    },
                                    "clarity": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 10,
                                        "description": "Score for clarity (1-10)",
                                    },
                                },
                                "required": ["correctness", "completeness", "clarity"],
                            },
                        }
                    ],
                    tool_choice={"type": "tool", "name": "evaluate_answer"},
                    messages=[{"role": "user", "content": prompt}],
                )

                # Extract tool call result
                for block in response.content:
                    if (
                        hasattr(block, "type")
                        and block.type == "tool_use"
                        and block.name == "evaluate_answer"
                    ):
                        return self._validate_scores(block.input)

                raise ValueError("No tool call found in response")

            except Exception as e:
                logger.warning(f"LLM judge attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.warning(
                        "All LLM judge attempts failed, returning default scores"
                    )
                    return JudgeScores(correctness=5, completeness=5, clarity=5)
                time.sleep(2)

        return JudgeScores(correctness=5, completeness=5, clarity=5)
