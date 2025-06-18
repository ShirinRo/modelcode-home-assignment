"""
QA Data Loader Module
Handles loading Q&A pairs from various sources.
"""

import json
import logging
from typing import List
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """Represents a question-answer pair for evaluation"""

    question: str
    reference_answer: str


class QADataLoader:
    """Handles loading Q&A pairs from various sources"""

    @staticmethod
    def load_from_github(
        repo_url: str = "https://api.github.com/repos/Modelcode-ai/grip_qa/contents",
    ) -> List[QAPair]:
        """Load Q&A pairs from the Grip QA GitHub repository"""
        qa_pairs = []

        try:
            logger.info(f"Loading Q&A pairs from GitHub: {repo_url}")
            response = requests.get(repo_url)
            response.raise_for_status()
            contents = response.json()

            # Group files by their number
            qa_files = {}
            for item in contents:
                if item["name"].endswith(".q.md") or item["name"].endswith(".a.md"):
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
                        question_response = requests.get(urls["question_url"])
                        question_response.raise_for_status()
                        question_text = question_response.text.strip()

                        answer_response = requests.get(urls["answer_url"])
                        answer_response.raise_for_status()
                        answer_text = answer_response.text.strip()

                        qa_pairs.append(
                            QAPair(question=question_text, reference_answer=answer_text)
                        )

                    except Exception as e:
                        logger.warning(f"Error processing Q&A pair {file_number}: {e}")
                        continue

            logger.info(f"Successfully loaded {len(qa_pairs)} Q&A pairs from GitHub")

        except Exception as e:
            logger.error(f"Error loading from GitHub: {e}")
            logger.info("Falling back to sample data")
            return QADataLoader.load_sample_data()

        return qa_pairs

    @staticmethod
    def load_sample_data() -> List[QAPair]:
        """Load sample Q&A pairs for testing"""
        return [
            QAPair(
                question="What does the Server class do?",
                reference_answer="The Server class is the main MCP server implementation that handles tool registration and execution for code Q&A functionality.",
            ),
            QAPair(
                question="How is the RAG system implemented?",
                reference_answer="The RAG system is implemented using a combination of code parsing, vector storage with ChromaDB, and semantic search to retrieve relevant code chunks for answering questions.",
            ),
            QAPair(
                question="How does the CodeParser parse Python files?",
                reference_answer="The CodeParser uses Python's AST module to parse files into logical chunks like classes, functions, and methods, extracting metadata and dependencies for each chunk.",
            ),
            QAPair(
                question="What is the purpose of the VectorStore class?",
                reference_answer="The VectorStore class handles vector storage and retrieval using ChromaDB, encoding code chunks as vectors for semantic search capabilities.",
            ),
            QAPair(
                question="How are code chunks created?",
                reference_answer="Code chunks are created by parsing Python AST nodes to extract logical code blocks like functions, classes, and methods, along with their metadata and dependencies.",
            ),
        ]

    @staticmethod
    def load_from_file(file_path: str) -> List[QAPair]:
        """Load Q&A pairs from a JSON file"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            qa_pairs = []
            for item in data:
                qa_pairs.append(
                    QAPair(
                        question=item["question"],
                        reference_answer=item["reference_answer"],
                    )
                )

            logger.info(f"Loaded {len(qa_pairs)} Q&A pairs from file: {file_path}")
            return qa_pairs

        except Exception as e:
            logger.error(f"Error loading from file {file_path}: {e}")
            raise
