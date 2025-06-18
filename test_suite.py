#!/usr/bin/env python3
"""
Comprehensive Test Suite for MCP Code QA Server
Tests all components of the system to ensure functionality and quality.
"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
import asyncio
from unittest.mock import patch, MagicMock

# Import our modules
from mcp_code_qa_server import RAGSystem, CodeParser, VectorStore, CodeChunk
from evaluation_script import QAEvaluator, QAPair, GripQALoader
from repo_analysis_agent import RepoAnalyzer, ArchitectureInfo, DependencyInfo


class TestCodeParser(unittest.TestCase):
    """Test the CodeParser functionality"""

    def setUp(self):
        self.parser = CodeParser()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def create_test_file(self, filename: str, content: str) -> str:
        """Create a test Python file"""
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, "w") as f:
            f.write(content)
        return file_path

    def test_simple_class_parsing(self):
        """Test parsing a simple class"""
        content = '''
class TestClass:
    """A test class"""
    
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        """Return the value"""
        return self.value
'''
        file_path = self.create_test_file("test_class.py", content)
        chunks = self.parser.parse_file(file_path)

        # Should have class chunk and method chunks
        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        method_chunks = [c for c in chunks if c.chunk_type == "method"]

        self.assertEqual(len(class_chunks), 1)
        self.assertEqual(class_chunks[0].name, "TestClass")
        self.assertEqual(class_chunks[0].docstring, "A test class")

        self.assertEqual(len(method_chunks), 2)  # __init__ and get_value
        method_names = [m.name for m in method_chunks]
        self.assertIn("__init__", method_names)
        self.assertIn("get_value", method_names)

    def test_function_parsing(self):
        """Test parsing standalone functions"""
        content = '''
import os
import sys

def hello_world():
    """Say hello"""
    print("Hello, World!")

def add_numbers(a, b):
    """Add two numbers"""
    return a + b
'''
        file_path = self.create_test_file("test_functions.py", content)
        chunks = self.parser.parse_file(file_path)

        import_chunks = [c for c in chunks if c.chunk_type == "import"]
        function_chunks = [c for c in chunks if c.chunk_type == "function"]

        self.assertEqual(len(import_chunks), 1)
        self.assertEqual(len(function_chunks), 2)

        function_names = [f.name for f in function_chunks]
        self.assertIn("hello_world", function_names)
        self.assertIn("add_numbers", function_names)

    def test_dependency_extraction(self):
        """Test dependency extraction"""
        content = """
def process_data(data):
    result = data.process()
    helper_function(result)
    return result.value
"""
        file_path = self.create_test_file("test_deps.py", content)
        chunks = self.parser.parse_file(file_path)

        function_chunk = [c for c in chunks if c.chunk_type == "function"][0]
        dependencies = function_chunk.dependencies or []

        self.assertIn("process", dependencies)
        self.assertIn("helper_function", dependencies)
        self.assertIn("value", dependencies)


class TestVectorStore(unittest.TestCase):
    """Test the VectorStore functionality"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = VectorStore(persist_directory=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_add_and_search_chunks(self):
        """Test adding chunks and searching"""
        chunks = [
            CodeChunk(
                content="def hello_world():\n    print('Hello!')",
                chunk_type="function",
                name="hello_world",
                file_path="test.py",
                start_line=1,
                end_line=2,
                docstring="Say hello",
            ),
            CodeChunk(
                content="class Calculator:\n    def add(self, a, b):\n        return a + b",
                chunk_type="class",
                name="Calculator",
                file_path="calc.py",
                start_line=1,
                end_line=3,
                docstring="A calculator class",
            ),
        ]

        self.vector_store.add_chunks(chunks)

        # Search for hello function
        results = self.vector_store.search("hello function", n_results=2)
        self.assertGreater(len(results), 0)

        # Search for calculator
        results = self.vector_store.search("calculator math", n_results=2)
        self.assertGreater(len(results), 0)


class TestRAGSystem(unittest.TestCase):
    """Test the complete RAG system"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.create_sample_repo()
        self.rag_system = RAGSystem(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def create_sample_repo(self):
        """Create a sample repository for testing"""
        # Main module
        main_content = '''
"""Main module for the application"""

from utils import helper_function
from models import DataProcessor

def main():
    """Main entry point"""
    processor = DataProcessor()
    result = processor.process_data([1, 2, 3])
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
'''

        # Utils module
        utils_content = '''
"""Utility functions"""

def helper_function(data):
    """Helper function to process data"""
    return [x * 2 for x in data]

def format_output(data):
    """Format data for output"""
    return ", ".join(map(str, data))
'''

        # Models module
        models_content = '''
"""Data models and processors"""

class DataProcessor:
    """Processes data using various algorithms"""
    
    def __init__(self):
        self.algorithm = "default"
    
    def process_data(self, data):
        """Process the input data"""
        from utils import helper_function
        return helper_function(data)
    
    def set_algorithm(self, algorithm):
        """Set the processing algorithm"""
        self.algorithm = algorithm
'''

        # Create files
        with open(os.path.join(self.temp_dir, "main.py"), "w") as f:
            f.write(main_content)

        with open(os.path.join(self.temp_dir, "utils.py"), "w") as f:
            f.write(utils_content)

        with open(os.path.join(self.temp_dir, "models.py"), "w") as f:
            f.write(models_content)

    def test_repository_indexing(self):
        """Test indexing a repository"""
        self.rag_system.index_repository()

        # Check that chunks were created
        self.assertGreater(len(self.rag_system.chunks), 0)

        # Check different chunk types exist
        chunk_types = {chunk.chunk_type for chunk in self.rag_system.chunks}
        self.assertIn("function", chunk_types)
        self.assertIn("class", chunk_types)
        self.assertIn("import", chunk_types)

    def test_question_answering(self):
        """Test answering questions about the code"""
        self.rag_system.index_repository()

        # Test questions
        questions = [
            "What does the DataProcessor class do?",
            "How is the main function implemented?",
            "What utility functions are available?",
        ]

        for question in questions:
            answer = self.rag_system.answer_question(question)
            self.assertIsInstance(answer, str)
            self.assertGreater(len(answer), 10)  # Should have substantial content


class TestQAEvaluator(unittest.TestCase):
    """Test the QA evaluation system"""

    def setUp(self):
        self.evaluator = QAEvaluator()

    def test_evaluation_metrics(self):
        """Test individual evaluation metrics"""
        question = "What does the function do?"
        reference = "The function processes data and returns the result"
        generated = (
            "This function takes data as input and processes it to return a result"
        )

        result = self.evaluator.evaluate_single(question, reference, generated)

        # Check that all metrics are computed
        self.assertIsInstance(result.semantic_similarity, float)
        self.assertIsInstance(result.rouge_l, float)
        self.assertIsInstance(result.overall_score, float)

        # Check reasonable ranges
        self.assertGreaterEqual(result.semantic_similarity, 0.0)
        self.assertLessEqual(result.semantic_similarity, 1.0)
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 1.0)

    def test_perfect_match(self):
        """Test evaluation with perfect match"""
        question = "Test question"
        answer = "This is the perfect answer"

        result = self.evaluator.evaluate_single(question, answer, answer)

        # Perfect match should have high scores
        self.assertGreater(result.semantic_similarity, 0.9)
        self.assertGreater(result.overall_score, 0.8)


class TestGripQALoader(unittest.TestCase):
    """Test the Grip QA dataset loader"""

    def test_sample_data_loading(self):
        """Test loading sample Q&A data"""
        qa_pairs = GripQALoader.load_sample_data()

        self.assertGreater(len(qa_pairs), 0)

        for qa_pair in qa_pairs:
            self.assertIsInstance(qa_pair.question, str)
            self.assertIsInstance(qa_pair.reference_answer, str)
            self.assertGreater(len(qa_pair.question), 0)
            self.assertGreater(len(qa_pair.reference_answer), 0)


class TestRepoAnalyzer(unittest.TestCase):
    """Test the repository analyzer"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_repo()
        self.analyzer = RepoAnalyzer(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def create_test_repo(self):
        """Create a test repository with various patterns"""
        # Create requirements.txt
        requirements_content = """
numpy==1.24.0
pandas>=1.5.0
flask==2.3.0
pytest>=7.0.0
"""

        with open(os.path.join(self.temp_dir, "requirements.txt"), "w") as f:
            f.write(requirements_content)

        # Create main application
        app_content = '''
"""Main application using Flask"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

class DataService:
    """Service for data processing"""
    
    def __init__(self):
        self.data = []
    
    def process_data(self, input_data):
        """Process input data using numpy"""
        arr = np.array(input_data)
        return arr.mean()

@app.route('/api/data', methods=['POST'])
def process_request():
    """API endpoint for data processing"""
    service = DataService()
    data = request.get_json()
    result = service.process_data(data['values'])
    return jsonify({'result': result})

if __name__ == "__main__":
    app.run(debug=True)
'''

        with open(os.path.join(self.temp_dir, "app.py"), "w") as f:
            f.write(app_content)

    def test_dependency_analysis(self):
        """Test dependency analysis"""
        analysis = self.analyzer.analyze_repository()

        # Should find external dependencies
        self.assertIn("numpy", analysis.dependencies.external_imports)
        self.assertIn("pandas", analysis.dependencies.external_imports)
        self.assertIn("flask", analysis.dependencies.external_imports)

        # Should categorize dependencies
        self.assertIn("web_frameworks", analysis.dependencies.dependency_categories)
        self.assertIn("data_science", analysis.dependencies.dependency_categories)

    def test_architecture_analysis(self):
        """Test architecture analysis"""
        analysis = self.analyzer.analyze_repository()

        # Should find main modules
        self.assertGreater(len(analysis.architecture.main_modules), 0)

        # Should find classes
        self.assertGreater(len(analysis.architecture.key_classes), 0)

        # Should identify entry points
        self.assertIn("app.py", analysis.architecture.entry_points)


class IntegrationTest(unittest.TestCase):
    """Integration tests for the complete system"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.create_complex_repo()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def create_complex_repo(self):
        """Create a complex repository for integration testing"""
        # Create package structure
        package_dir = os.path.join(self.temp_dir, "mypackage")
        os.makedirs(package_dir)

        # Package __init__.py
        init_content = '''
"""A sample package for testing"""

from .core import MainProcessor
from .utils import helper_functions

__version__ = "1.0.0"
__all__ = ["MainProcessor", "helper_functions"]
'''

        # Core module
        core_content = '''
"""Core processing functionality"""

from typing import List, Dict, Any
import json
from .utils import validate_input

class MainProcessor:
    """Main processing class implementing Strategy pattern"""
    
    def __init__(self, strategy: str = "default"):
        self.strategy = strategy
        self.processors = {
            "default": self._default_process,
            "advanced": self._advanced_process
        }
    
    def process(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process data using selected strategy"""
        if not validate_input(data):
            raise ValueError("Invalid input data")
        
        processor = self.processors.get(self.strategy, self._default_process)
        return processor(data)
    
    def _default_process(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Default processing implementation"""
        return {"result": len(data), "strategy": "default"}
    
    def _advanced_process(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Advanced processing implementation"""
        total_items = sum(len(item) for item in data)
        return {"result": total_items, "strategy": "advanced"}

def create_processor(strategy: str) -> MainProcessor:
    """Factory function for creating processors"""
    return MainProcessor(strategy)
'''

        # Utils module
        utils_content = '''
"""Utility functions and helpers"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def validate_input(data: List[Dict[str, Any]]) -> bool:
    """Validate input data structure"""
    if not isinstance(data, list):
        logger.error("Input must be a list")
        return False
    
    for item in data:
        if not isinstance(item, dict):
            logger.error("All items must be dictionaries")
            return False
    
    return True

class ConfigManager:
    """Singleton pattern for configuration management"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = {}
        return cls._instance
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self._config[key] = value
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)

def helper_functions():
    """Collection of helper functions"""
    return {
        "validate": validate_input,
        "config": ConfigManager()
    }
'''

        # Write files
        with open(os.path.join(package_dir, "__init__.py"), "w") as f:
            f.write(init_content)

        with open(os.path.join(package_dir, "core.py"), "w") as f:
            f.write(core_content)

        with open(os.path.join(package_dir, "utils.py"), "w") as f:
            f.write(utils_content)

        # Main entry point
        main_content = '''
#!/usr/bin/env python3
"""Main entry point for the application"""

import sys
import json
from mypackage import MainProcessor, helper_functions

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python main.py <strategy>")
        sys.exit(1)
    
    strategy = sys.argv[1]
    processor = MainProcessor(strategy)
    
    # Sample data
    test_data = [
        {"id": 1, "value": "test1"},
        {"id": 2, "value": "test2"}
    ]
    
    try:
        result = processor.process(test_data)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

        with open(os.path.join(self.temp_dir, "main.py"), "w") as f:
            f.write(main_content)

    def test_full_pipeline(self):
        """Test the complete analysis pipeline"""
        # Initialize RAG system
        rag_system = RAGSystem(self.temp_dir)
        rag_system.index_repository()

        # Test Q&A functionality
        questions = [
            "What design patterns are used in this code?",
            "How is the MainProcessor class implemented?",
            "What is the purpose of the ConfigManager class?",
            "How does the factory function work?",
        ]

        for question in questions:
            answer = rag_system.answer_question(question)
            self.assertIsInstance(answer, str)
            self.assertGreater(len(answer), 20)

        # Test repository analysis
        analyzer = RepoAnalyzer(self.temp_dir)
        analysis = analyzer.analyze_repository()

        # Verify analysis results
        self.assertIsInstance(analysis.summary, str)
        self.assertGreater(len(analysis.architecture.key_classes), 0)
        self.assertGreater(len(analysis.recommendations), 0)

        # Check for pattern detection
        patterns = analysis.architecture.design_patterns
        self.assertTrue(
            any("Singleton" in pattern for pattern in patterns)
            or any("Factory" in pattern for pattern in patterns)
        )


def run_performance_tests():
    """Run performance benchmarks"""
    import time

    print("\n" + "=" * 50)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 50)

    # Create a large test repository
    temp_dir = tempfile.mkdtemp()

    try:
        # Generate multiple files
        for i in range(50):
            content = f'''
"""Module {i}"""

class Class{i}:
    """Class number {i}"""
    
    def __init__(self):
        self.value = {i}
    
    def method_{i}(self, param):
        """Method {i}"""
        return param * {i}

def function_{i}():
    """Function {i}"""
    return {i} * 2
'''
            with open(os.path.join(temp_dir, f"module_{i}.py"), "w") as f:
                f.write(content)

        # Benchmark indexing
        start_time = time.time()
        rag_system = RAGSystem(temp_dir)
        rag_system.index_repository()
        indexing_time = time.time() - start_time

        print(f"Indexing 50 files: {indexing_time:.2f} seconds")
        print(f"Total chunks: {len(rag_system.chunks)}")
        print(f"Chunks per second: {len(rag_system.chunks)/indexing_time:.1f}")

        # Benchmark query performance
        test_questions = [
            "What does Class25 do?",
            "How is method_10 implemented?",
            "What functions are available in module_5?",
        ]

        query_times = []
        for question in test_questions:
            start_time = time.time()
            answer = rag_system.answer_question(question)
            query_time = time.time() - start_time
            query_times.append(query_time)
            print(f"Query '{question[:30]}...': {query_time:.3f} seconds")

        avg_query_time = sum(query_times) / len(query_times)
        print(f"Average query time: {avg_query_time:.3f} seconds")

    finally:
        shutil.rmtree(temp_dir)


class TestMCPIntegration(unittest.TestCase):
    """Test MCP protocol integration"""


def create_sample_qa_dataset():
    """Create a sample Q&A dataset for testing"""
    qa_pairs = [
        {
            "question": "What is the purpose of the RAGSystem class?",
            "answer": "The RAGSystem class implements a Retrieval Augmented Generation system for code Q&A. It combines code parsing, vector storage, and semantic search to answer questions about codebases.",
            "category": "architecture",
        },
        {
            "question": "How does the CodeParser extract dependencies?",
            "answer": "The CodeParser extracts dependencies by walking the AST nodes and identifying function calls, attribute access, and other references within the code.",
            "category": "implementation",
        },
        {
            "question": "What vector database is used for storage?",
            "answer": "The system uses ChromaDB as the vector database for storing and retrieving code chunk embeddings.",
            "category": "technology",
        },
        {
            "question": "How are code chunks created?",
            "answer": "Code chunks are created by parsing Python files using AST analysis to extract logical units like classes, functions, and methods with their metadata.",
            "category": "process",
        },
        {
            "question": "What evaluation metrics are used?",
            "answer": "The evaluation system uses semantic similarity, ROUGE-L, and BLEU scores to measure answer quality against reference answers.",
            "category": "evaluation",
        },
    ]

    return qa_pairs


def run_integration_test():
    """Run a full integration test"""
    print("\n" + "=" * 50)
    print("INTEGRATION TEST")
    print("=" * 50)

    # Create temporary repository
    temp_dir = tempfile.mkdtemp()

    try:
        # Create the main MCP server file as test subject
        server_content = '''
"""Simple test server for integration testing"""

import asyncio
from typing import List, Dict, Any

class SimpleServer:
    """A simple server implementation"""
    
    def __init__(self, name: str):
        self.name = name
        self.clients = []
    
    async def start(self):
        """Start the server"""
        print(f"Starting server: {self.name}")
        return True
    
    def add_client(self, client_id: str):
        """Add a client to the server"""
        self.clients.append(client_id)
        print(f"Added client: {client_id}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status"""
        return {
            "name": self.name,
            "clients": len(self.clients),
            "running": True
        }

async def main():
    """Main entry point"""
    server = SimpleServer("test-server")
    await server.start()
    
    server.add_client("client-1")
    server.add_client("client-2")
    
    status = server.get_status()
    print(f"Server status: {status}")

if __name__ == "__main__":
    asyncio.run(main())
'''

        with open(os.path.join(temp_dir, "test_server.py"), "w") as f:
            f.write(server_content)

        print("1. Testing RAG System...")
        rag_system = RAGSystem(temp_dir)
        rag_system.index_repository()
        print(f"   Indexed {len(rag_system.chunks)} chunks")

        print("2. Testing Q&A functionality...")
        test_questions = [
            "What does the SimpleServer class do?",
            "How is the start method implemented?",
            "What is the main entry point?",
        ]

        for question in test_questions:
            answer = rag_system.answer_question(question)
            print(f"   Q: {question}")
            print(f"   A: {answer[:100]}...")

        print("3. Testing evaluation system...")
        qa_pairs = [QAPair(q, "Test answer", "test") for q in test_questions]
        evaluator = QAEvaluator()
        results = evaluator.evaluate_batch(qa_pairs, rag_system)

        avg_score = sum(r.overall_score for r in results) / len(results)
        print(f"   Average evaluation score: {avg_score:.3f}")

        print("4. Testing repository analysis...")
        analyzer = RepoAnalyzer(temp_dir)
        analysis = analyzer.analyze_repository()
        print(f"   Found {len(analysis.architecture.key_classes)} classes")
        print(f"   Generated {len(analysis.recommendations)} recommendations")

        print("\n✅ Integration test completed successfully!")

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        raise
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("MCP Code QA Server - Test Suite")
    print("=" * 50)

    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(argv=[""], exit=False, verbosity=2)

    # Run performance tests
    run_performance_tests()

    # Run integration test
    run_integration_test()

    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED")
    print("=" * 50)
