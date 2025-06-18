#!/usr/bin/env python3
"""
Retrieval Augmented Generation (RAG).
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

# Third-party imports
import ast
import tiktoken
from sentence_transformers import SentenceTransformer
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a logical chunk of code with metadata"""

    content: str
    chunk_type: str  # 'class', 'function', 'method', 'module', 'import'
    name: str
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    parent_class: Optional[str] = None
    dependencies: Optional[List[str]] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class CodeParser:
    """Parse Python code into logical chunks"""

    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def parse_file(self, file_path: str) -> List[CodeChunk]:
        """Parse a Python file into logical code chunks"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            chunks = []
            lines = content.split("\n")

            # Extract imports first
            imports = self._extract_imports(tree, lines, file_path)
            chunks.extend(imports)

            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    chunk = self._extract_class(node, lines, file_path)
                    chunks.append(chunk)

                    # Extract methods within the class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_chunk = self._extract_method(
                                item, lines, file_path, node.name
                            )
                            chunks.append(method_chunk)

                elif isinstance(node, ast.FunctionDef):
                    # Only top-level functions (not methods)
                    if not any(
                        isinstance(parent, ast.ClassDef)
                        for parent in ast.walk(tree)
                        if hasattr(parent, "body")
                        and node in getattr(parent, "body", [])
                    ):
                        chunk = self._extract_function(node, lines, file_path)
                        chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return []

    def _extract_imports(
        self, tree: ast.AST, lines: List[str], file_path: str
    ) -> List[CodeChunk]:
        """Extract import statements"""
        chunks = []
        import_lines = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines.append(node.lineno)

        if import_lines:
            start_line = min(import_lines)
            end_line = max(import_lines)
            content = "\n".join(lines[start_line - 1 : end_line])

            chunk = CodeChunk(
                content=content,
                chunk_type="import",
                name="imports",
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
            )
            chunks.append(chunk)

        return chunks

    def _extract_class(
        self, node: ast.ClassDef, lines: List[str], file_path: str
    ) -> CodeChunk:
        """Extract a class definition"""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        content = "\n".join(lines[start_line - 1 : end_line])

        docstring = ast.get_docstring(node)
        dependencies = self._extract_dependencies(node)

        return CodeChunk(
            content=content,
            chunk_type="class",
            name=node.name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            docstring=docstring,
            dependencies=dependencies,
        )

    def _extract_function(
        self, node: ast.FunctionDef, lines: List[str], file_path: str
    ) -> CodeChunk:
        """Extract a function definition"""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        content = "\n".join(lines[start_line - 1 : end_line])

        docstring = ast.get_docstring(node)
        dependencies = self._extract_dependencies(node)

        return CodeChunk(
            content=content,
            chunk_type="function",
            name=node.name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            docstring=docstring,
            dependencies=dependencies,
        )

    def _extract_method(
        self, node: ast.FunctionDef, lines: List[str], file_path: str, parent_class: str
    ) -> CodeChunk:
        """Extract a method definition"""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        content = "\n".join(lines[start_line - 1 : end_line])

        docstring = ast.get_docstring(node)
        dependencies = self._extract_dependencies(node)

        return CodeChunk(
            content=content,
            chunk_type="method",
            name=node.name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            docstring=docstring,
            parent_class=parent_class,
            dependencies=dependencies,
        )

    def _extract_dependencies(self, node: ast.AST) -> List[str]:
        """Extract dependencies (function calls, attribute access) from a node"""
        dependencies = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    dependencies.add(child.func.attr)
            elif isinstance(child, ast.Attribute):
                dependencies.add(child.attr)

        return list(dependencies)


class VectorStore:
    """Vector storage and retrieval using ChromaDB"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="code_chunks", metadata={"hnsw:space": "cosine"}
        )
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def add_chunks(self, chunks: List[CodeChunk]):
        """Add code chunks to the vector store"""
        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            # Create searchable text combining code and metadata
            searchable_text = f"{chunk.chunk_type}: {chunk.name}\n"
            if chunk.docstring:
                searchable_text += f"Docstring: {chunk.docstring}\n"
            searchable_text += f"Code:\n{chunk.content}"

            documents.append(searchable_text)

            metadata = {
                "chunk_type": chunk.chunk_type,
                "name": chunk.name,
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "parent_class": chunk.parent_class or "",
                "dependencies": json.dumps(chunk.dependencies),
            }
            metadatas.append(metadata)
            ids.append(f"{chunk.file_path}:{chunk.start_line}:{chunk.name}")

        if documents:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant code chunks"""
        results = self.collection.query(query_texts=[query], n_results=n_results)

        if (
            not results["documents"]
            or not results["metadatas"]
            or not results["distances"]
        ):
            return []

        return [
            {"content": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def clear(self):
        """Clear all data from the vector store"""
        self.client.delete_collection("code_chunks")
        self.collection = self.client.get_or_create_collection(
            name="code_chunks", metadata={"hnsw:space": "cosine"}
        )


class RAGSystem:
    """Retrieval Augmented Generation system for code Q&A"""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.parser = CodeParser()
        self.vector_store = VectorStore()
        self.chunks = []

    def index_repository(self):
        """Index the entire repository"""
        logger.info(f"Indexing repository: {self.repo_path}")

        # Clear existing data
        self.vector_store.clear()
        self.chunks = []

        # Find all Python files
        python_files = list(self.repo_path.rglob("*.py"))
        logger.info(f"Found {len(python_files)} Python files")

        # Parse each file
        for file_path in python_files:
            try:
                file_chunks = self.parser.parse_file(str(file_path))
                self.chunks.extend(file_chunks)
                logger.info(f"Parsed {len(file_chunks)} chunks from {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        # Add to vector store
        if self.chunks:
            self.vector_store.add_chunks(self.chunks)
            logger.info(f"Indexed {len(self.chunks)} total chunks")

    def answer_question(self, question: str) -> str:
        """Answer a question about the codebase"""
        # Retrieve relevant chunks
        relevant_chunks = self.vector_store.search(question, n_results=10)

        if not relevant_chunks:
            return "I couldn't find relevant code to answer your question."

        # Build context from retrieved chunks
        context_parts = []
        for chunk in relevant_chunks[:5]:  # Use top 5 most relevant
            metadata = chunk["metadata"]
            content = chunk["content"]

            context_part = f"""
File: {metadata['file_path']}
Type: {metadata['chunk_type']}
Name: {metadata['name']}
{f"Class: {metadata['parent_class']}" if metadata['parent_class'] else ""}

{content}
---
"""
            context_parts.append(context_part)

        context = "\n".join(context_parts)

        # Generate answer (simple template-based approach)
        # In a production system, you'd use an LLM here
        answer = self._generate_answer(question, context, relevant_chunks)

        return answer

    def _generate_answer(self, question: str, context: str, chunks: List[Dict]) -> str:
        """Generate an answer based on the question and context"""
        # Simple rule-based answer generation
        # In practice, you'd use an LLM like OpenAI GPT or similar

        question_lower = question.lower()

        if "what does" in question_lower and "class" in question_lower:
            # Find class-related chunks
            class_chunks = [c for c in chunks if c["metadata"]["chunk_type"] == "class"]
            if class_chunks:
                chunk = class_chunks[0]
                metadata = chunk["metadata"]
                return f"The class '{metadata['name']}' is defined in {metadata['file_path']}. Here's its implementation:\n\n{chunk['content']}"

        elif "how is" in question_lower and "implement" in question_lower:
            # Look for implementation details
            relevant_chunk = chunks[0] if chunks else None
            if relevant_chunk:
                metadata = relevant_chunk["metadata"]
                return f"The implementation can be found in {metadata['file_path']}:\n\n{relevant_chunk['content']}"

        elif "how does" in question_lower and "method" in question_lower:
            # Look for method usage
            method_chunks = [
                c
                for c in chunks
                if c["metadata"]["chunk_type"] in ["method", "function"]
            ]
            if method_chunks:
                chunk = method_chunks[0]
                metadata = chunk["metadata"]
                return f"The method '{metadata['name']}' works as follows:\n\n{chunk['content']}"

        # Default response with most relevant chunk
        if chunks:
            chunk = chunks[0]
            metadata = chunk["metadata"]
            return f"Based on the code analysis, here's the relevant information from {metadata['file_path']}:\n\n{chunk['content']}"

        return "I couldn't generate a specific answer for your question based on the available code."
