# Modelcode GenAI Coding Exercise - Shirin Robinov

---

## 1. Setup

**a. Create virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

**b. Install requirements:**

```bash
pip install -r requirements.txt
```

**c. Add Anthropic API key:**

Create a `.env` file in the project root with:

```
ANTHROPIC_API_KEY=sk-...
```

---

## 2. Analyzer

**How to run:**

```bash
python -m analyzer.analyzer_agent --repo-path grip-no-tests
```

**What it does:**

* Connects to your MCP server and uses it as a tool.
* Analyzes the codebase and generates a professional markdown report covering:

  * Architecture
  * Dependencies
  * Design patterns
* Uses an LLM to synthesize insights and structure the report.
* Output is saved to `analyzer/results/analysis_report.md`.

**Structure:**

* `analyzer_agent.py`: Main entry point and orchestration.
* `analysis_reducer.py`: Summarizes sections and generates the final report using Claude.

---

## 3. Evaluation

**How to run:**

```bash
python -m evaluation.evaluator --repo-path grip-no-tests
```

**What it does:**

* Automatically evaluates your QA system on a set of 10 reference Q\&A pairs (downloaded from GitHub).
* Measures quality using:

  * Semantic similarity & ROUGE-L (NLP metrics)
  * LLM-based judge scoring (correctness, completeness, clarity)
* Saves results and statistics to `evaluation/evaluation_report.json`.

**Structure:**

* `evaluator.py`: Main evaluation workflow.
* `nlp_evaluator.py`: Handles NLP scoring.
* `llm_judge_evaluator.py`: Handles LLM judge scoring.
* `evaluation_reporter.py`: Compiles and reports results.

---

## 4. MCP Section

**General explanation:**

* The MCP section implements a server-client architecture for codebase Q\&A.
* **Server (`mcp_server.py`)** exposes two tools over MCP:

  * `index_repository`: Indexes a Python repo for questions.
  * `ask_question`: Answers natural language questions using RAG and LLM.
* **Client (`mcp_client.py`)** connects to the server and mediates between the LLM (Claude) and the available MCP tools.
* **RAG system (`rag_system.py`)** handles chunking, embedding, vector search, and builds structured answers for any code question.

---
