# MCP Code QA Server - Setup and Usage Instructions

## Overview

This project implements a Model Context Protocol (MCP) server for code Q&A using Retrieval Augmented Generation (RAG). The system can answer natural language questions about local Python repositories by parsing code into logical chunks, storing them in a vector database, and retrieving relevant information to answer questions.

## Components

1. **MCP Server** (`mcp_server.py`) - Main server implementing the MCP protocol for code Q&A
2. **Evaluation Script** (`evaluation.py`) - Measures system quality against reference Q&A pairs
3. **Repository Analysis Agent** (`repo_analyzer.py`) - Analyzes repositories and generates comprehensive reports

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download the project files**
   ```bash
   mkdir mcp-code-qa
   cd mcp-code-qa
   # Place all Python files in this directory
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data (required for evaluation)**
   ```python
   python -c "import nltk; nltk.download('punkt')"
   ```

## Usage

### 1. MCP Server Setup

The MCP server can be used as a standalone tool or integrated with MCP-compatible clients.

#### Standalone Usage

```python
from mcp_server import RAGSystem

# Initialize and index a repository
rag_system = RAGSystem("/path/to/your/python/repo")
rag_system.index_repository()

# Ask questions
answer = rag_system.answer_question("What does the Server class do?")
print(answer)
```

#### Running as MCP Server

```bash
python mcp_server.py
```

The server will start and listen for MCP protocol messages via stdio.

#### Available Tools

- `index_repository`: Index a Python repository for Q&A
- `ask_question`: Ask natural language questions about the indexed code
- `get_repository_info`: Get statistics about the indexed repository

### 2. Evaluation Script

Evaluate the system's performance against reference Q&A pairs:

```bash
# Using sample data
python evaluation.py --repo-path /path/to/test/repo --qa-source sample

# Using GitHub Q&A dataset (when available)
python evaluation.py --repo-path /path/to/test/repo --qa-source github

# With custom output file
python evaluation.py --repo-path /path/to/test/repo --output my_evaluation.json --verbose
```

#### Evaluation Metrics

- **Semantic Similarity**: Cosine similarity between sentence embeddings
- **ROUGE-L**: Longest common subsequence based metric
- **BLEU Score**: N-gram overlap metric
- **Overall Score**: Weighted combination of all metrics

### 3. Repository Analysis Agent

Generate comprehensive repository analysis reports:

```bash
# Generate both markdown and JSON reports
python repo_analyzer.py /path/to/repo

# Generate only markdown report
python repo_analyzer.py /path/to/repo --output-format markdown

# Specify output directory
python repo_analyzer.py /path/to/repo --output-dir ./reports --verbose
```

#### Report Contents

- **Architecture Overview**: Modules, packages, entry points, key classes/functions
- **Dependencies**: External libraries, categorization, potential issues
- **Code Quality**: Documentation coverage, complexity metrics
- **Design Patterns**: Identified patterns in the codebase
- **Recommendations**: Actionable suggestions for improvement

## Configuration

### Vector Store Configuration

The system uses ChromaDB by default. You can configure the storage location:

```python
# In mcp_server.py, modify VectorStore initialization
vector_store = VectorStore(persist_directory="./custom_chroma_db")
```

### Chunk Size and Retrieval

Adjust the number of retrieved chunks for answering questions:

```python
# In RAGSystem.answer_question method
relevant_chunks = self.vector_store.search(question, n_results=10)  # Increase for more context
```

### Evaluation Thresholds

Modify score interpretation thresholds in `evaluation.py`:

```python
# In EvaluationReporter.print_summary
if avg_overall >= 0.8:  # Excellent threshold
    print("🟢 Excellent performance!")
```

## Testing the System

### Quick Test with Sample Repository

1. **Download the provided test repository**
   ```bash
   # Download from the Google Drive link provided in the exercise
   # Extract to a local directory
   ```

2. **Run evaluation**
   ```bash
   python evaluation.py --repo-path ./test-repo --qa-source sample --verbose
   ```

3. **Generate analysis report**
   ```bash
   python repo_analyzer.py ./test-repo --output-dir ./reports
   ```

### Custom Testing

1. **Index your own repository**
   ```python
   from mcp_server import RAGSystem
   
   rag = RAGSystem("/path/to/your/repo")
   rag.index_repository()
   
   # Test with questions
   questions = [
       "What is the main entry point?",
       "How is error handling implemented?",
       "What external libraries are used?"
   ]
   
   for q in questions:
       print(f"Q: {q}")
       print(f"A: {rag.answer_question(q)}\n")
   ```

## Architecture Details

### Code Parsing Strategy

The system parses Python code into logical chunks:

- **Classes**: Complete class definitions with metadata
- **Functions**: Top-level functions with dependencies
- **Methods**: Class methods linked to parent classes
- **Imports**: Import statements grouped by file
- **Modules**: File-level organization

### Vector Storage

- Uses sentence-transformers for encoding
- ChromaDB for persistent vector storage
- Cosine similarity for retrieval
- Metadata filtering capabilities

### RAG Pipeline

1. **Indexing**: Parse code → Extract chunks → Generate embeddings → Store vectors
2. **Retrieval**: Query → Search vectors → Rank by similarity → Select top chunks
3. **Generation**: Combine retrieved chunks → Apply templates → Generate answer

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **ChromaDB Permission Issues**
   ```bash
   # Change the persist directory
   chmod 755 ./chroma_db
   ```

3. **Memory Issues with Large Repositories**
   ```python
   # Reduce chunk size or implement batching
   # In CodeParser, limit the number of chunks processed
   ```

4. **Poor Answer Quality**
   - Increase the number of retrieved chunks
   - Improve the answer generation templates
   - Add more sophisticated filtering

### Performance Optimization

1. **Faster Indexing**
   - Use parallel processing for file parsing
   - Implement incremental indexing
   - Cache parsed results

2. **Better Retrieval**
   - Experiment with different embedding models
   - Implement hybrid search (vector + keyword)
   - Add query expansion techniques

3. **Improved Generation**
   - Integrate with LLM APIs (OpenAI, Anthropic)
   - Implement better template systems
   - Add context-aware generation

## Extension Ideas

### Integration with LLMs

```python
# Add to RAGSystem class
def _generate_answer_with_llm(self, question: str, context: str) -> str:
    # Integrate with OpenAI API or similar
    prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    # Call LLM API
    return llm_response
```

### Advanced Chunking

- Support for other languages (JavaScript, Java)
- Semantic chunking based on code functionality
- Hierarchical chunking with multiple granularities

### Enhanced Evaluation

- Human evaluation framework
- Domain-specific metrics
- Comparative evaluation against other systems

## Contributing

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Include docstrings for new methods
4. Add tests for new functionality
5. Update documentation for new features

## License

This project is provided as an educational example for the Modelcode GenAI coding exercise.
