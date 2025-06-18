#!/usr/bin/env python3
"""
Improved Retrieval Augmented Generation (RAG) system for code repositories.
Maintains API compatibility with existing MCP server.
"""

import os
from pathlib import Path
from typing import List, cast
import logging

# Third-party imports
from langchain.text_splitter import PythonCodeTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, BaseMessage
from pydantic import SecretStr
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """Enhanced RAG system for code Q&A with backward compatibility"""

    def __init__(self, repo_path: str):
        """
        Initialize the RAG system.

        Args:
            repo_path: Path to the repository to index
        """
        self.repo_path = Path(repo_path)
        self.repo_description = ""

        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.text_splitter = PythonCodeTextSplitter(
            chunk_size=1200,
            chunk_overlap=250,
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store = None

        if not anthropic_api_key:
            logger.error("No Anthropic API key provided.")
            self.llm = None
        else:
            self.llm = ChatAnthropic(
                model_name="claude-3-5-sonnet-20241022",
                api_key=SecretStr(anthropic_api_key),
                temperature=0.1,
                max_tokens_to_sample=1500,
                timeout=45.0,
                stop=None,
            )

        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert Python code analyst. Your task is to answer questions about a codebase using the provided code snippets.

Retrieved Code Context:
{context}

Question: {question}

Instructions for your response:
1. Analyze the provided code snippets carefully
2. Give a direct, clear answer to the question
3. Reference specific files, functions, or classes when relevant
4. Include relevant code examples if they help explain your answer
5. If the code context doesn't fully answer the question, clearly state what information is missing
6. Be precise and technical when discussing code implementation details

Structure your response as:
**Answer:** [Direct answer to the question]

**Code Analysis:** [Analysis of relevant code with file references]

**Details:** [Implementation specifics, if relevant]

**Limitations:** [Note any missing information, if applicable]

Response:""",
        )

    def _generate_repo_description(self, documents: List[Document]) -> str:
        """Generate a simple description of the repository"""
        file_count = len(documents)
        directories = set()

        for doc in documents:
            file_path = doc.metadata.get("file_path", "")
            if "/" in file_path:
                directories.add(file_path.split("/")[0])

        desc = f"Python repository with {file_count} files"
        if directories:
            desc += f" in directories: {', '.join(sorted(directories))}"

        return desc

    def index_repository(self):
        """Index the entire repository - maintains original API"""
        logger.info(f"Indexing repository: {self.repo_path}")

        # Find all Python files
        python_files = list(self.repo_path.rglob("*.py"))
        logger.info(f"Found {len(python_files)} Python files")

        documents = []
        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                if not content.strip():
                    continue

                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_path": str(file_path.relative_to(self.repo_path)),
                        "directory": str(file_path.relative_to(self.repo_path).parent),
                    },
                )
                documents.append(doc)

            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        if not documents:
            logger.warning("No documents to index")
            return

        self.repo_description = self._generate_repo_description(documents)
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        # Create vector store
        persist_dir = "./chroma_db"
        if os.path.exists(persist_dir):
            # Clean restart for consistency
            import shutil

            shutil.rmtree(persist_dir)

        self.vector_store = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir,
        )
        self.vector_store.persist()

        logger.info(f"Indexed {len(all_chunks)} chunks successfully")

    def _get_enhanced_context(self, question: str, k: int = 6) -> str:
        """Get enhanced context with better formatting"""
        if not self.vector_store:
            return "No repository indexed."

        try:
            # Try similarity search with score threshold first
            docs_and_scores = self.vector_store.similarity_search_with_score(
                question, k=k
            )

            # Filter by relevance score (lower is better for cosine distance)
            relevant_docs = [
                (doc, score) for doc, score in docs_and_scores if score < 0.8
            ]

            if not relevant_docs:
                # Fallback to regular search if no docs pass threshold
                docs = self.vector_store.similarity_search(question, k=k)
                relevant_docs = [(doc, 0.0) for doc in docs]

        except Exception as e:
            logger.warning(f"Search error, using fallback: {e}")
            docs = self.vector_store.similarity_search(question, k=k)
            relevant_docs = [(doc, 0.0) for doc in docs]

        if not relevant_docs:
            return "No relevant code found."

        # Format context with clear structure
        context_parts = []
        seen_files = set()

        for i, (doc, score) in enumerate(relevant_docs[:k], 1):
            file_path = doc.metadata.get("file_path", "unknown")

            # Add file header only once per file
            file_header = ""
            if file_path not in seen_files:
                seen_files.add(file_path)
                file_header = f"\n=== File: {file_path} ===\n"

            context_part = f"""{file_header}
--- Code Snippet {i} ---
{doc.page_content.strip()}
"""
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def answer_question(self, question: str) -> str:
        if not self.vector_store:
            return (
                "No repository has been indexed yet. Please index a repository first."
            )

        if not self.llm:
            return (
                "No LLM available. Please check your Anthropic API key configuration."
            )

        try:
            context = self._get_enhanced_context(question)
            formatted_prompt = self.qa_prompt.format(context=context, question=question)

            messages = [cast(BaseMessage, HumanMessage(content=formatted_prompt))]
            response = self.llm(messages)
            answer = response.content

            if not isinstance(answer, str):
                if isinstance(answer, list):
                    answer = "\n".join(
                        str(item) if isinstance(item, str) else str(item)
                        for item in answer
                    )
                else:
                    answer = str(answer)

            if self.vector_store:
                docs = self.vector_store.similarity_search(question, k=5)
                sources = set()
                for doc in docs:
                    if "file_path" in doc.metadata:
                        sources.add(doc.metadata["file_path"])

                if sources:
                    answer += f"\n\n**Sources:** {', '.join(sorted(sources))}"

            return answer

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Error processing question: {str(e)}"
