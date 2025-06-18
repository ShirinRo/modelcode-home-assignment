#!/usr/bin/env python3
"""
Simplified RAG System for Code Q&A
Uses LangChain's PythonCodeTextSplitter, ChromaDB, and HuggingFace embeddings
with Claude LLM integration for answer generation.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Third-party imports
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pydantic import SecretStr
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """Simplified RAG system for code Q&A using LangChain components"""

    def __init__(self, repo_path: str, anthropic_api_key: Optional[str] = None):
        """
        Initialize the RAG system.

        Args:
            repo_path: Path to the repository to index
            anthropic_api_key: Anthropic API key (optional, uses env var if not provided)
        """
        self.repo_path = Path(repo_path)
        self.chunks = []  # Keep for compatibility with MCP server
        anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

        # Initialize text splitter
        self.text_splitter = PythonCodeTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialize vector store
        self.vector_store = None

        # Initialize Claude LLM
        anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            logger.warning("No Anthropic API key provided. Using mock LLM for testing.")
            self.llm = MockLLM()
        else:
            self.llm = ChatAnthropic(
                model_name="claude-3-5-sonnet-20240620",
                api_key=SecretStr(anthropic_api_key or "anthropic_api_key"),
                temperature=0.3,
                max_tokens_to_sample=1024,
                timeout=30.0,
                stop=None,
            )

        # Create QA prompt template
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful code assistant. Use the following code snippets to answer the question about the codebase.

Context:
{context}

Question: {question}

Please provide a clear and concise answer based on the code provided. If the code doesn't contain enough information to fully answer the question, acknowledge what you can determine and what information is missing.

Answer:""",
        )

    def index_repository(self):
        """Index the entire repository"""
        logger.info(f"Indexing repository: {self.repo_path}")

        # Find all Python files
        python_files = list(self.repo_path.rglob("*.py"))
        logger.info(f"Found {len(python_files)} Python files")

        # Create documents from Python files
        documents = []

        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Skip empty files
                if not content.strip():
                    continue

                # Create document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_path": str(file_path.relative_to(self.repo_path)),
                    },
                )
                documents.append(doc)

            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        if not documents:
            logger.warning("No documents to index")
            return

        # Split documents into chunks
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            # Add file metadata to each chunk
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")

        # Store chunks for compatibility with MCP server
        self.chunks = all_chunks

        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db",
        )
        self.vector_store.persist()

        logger.info(f"Indexed {len(all_chunks)} chunks successfully")

    def answer_question(self, question: str) -> str:
        """Answer a question about the codebase"""
        if not self.vector_store:
            return (
                "No repository has been indexed yet. Please index a repository first."
            )

        try:
            # Create retrieval QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": self.qa_prompt},
                return_source_documents=True,
            )

            # Get answer
            result = qa_chain({"query": question})
            answer = result["result"]

            # Add source information
            if result.get("source_documents"):
                sources = set()
                for doc in result["source_documents"]:
                    if "file_path" in doc.metadata:
                        sources.add(doc.metadata["file_path"])

                if sources:
                    answer += f"\n\nSources: {', '.join(sorted(sources))}"

            return answer

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Error processing question: {str(e)}"


class MockLLM:
    """Mock LLM for testing without API token"""

    def __call__(self, prompt: str) -> str:
        """Simple mock response generator"""
        if isinstance(prompt, list):
            prompt = "Yes"

        # Extract context and question from prompt
        if "Context:" in prompt and "Question:" in prompt:
            context_start = prompt.find("Context:") + len("Context:")
            question_start = prompt.find("Question:")
            context = prompt[context_start:question_start].strip()[:200]

            return f"Based on the provided code context, I can see this relates to: {context}... [Note: This is a mock response. Please provide an Anthropic API key for actual Claude responses.]"

        return "Mock LLM response: Please provide an Anthropic API key for actual responses."

    @property
    def _llm_type(self) -> str:
        return "mock"


# For backward compatibility with imports
CodeChunk = None  # MCP server checks for this attribute


# Usage example in comments
"""
Usage Example:

# With Anthropic API key
rag_system = RAGSystem("/path/to/repo", anthropic_api_key="your-api-key")

# Or set environment variable
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"
rag_system = RAGSystem("/path/to/repo")

# Index repository
rag_system.index_repository()

# Ask questions
answer = rag_system.answer_question("How does the authentication system work?")
"""
