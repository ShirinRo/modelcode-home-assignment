#!/usr/bin/env python3
"""
Comprehensive Repository Analysis Agent
Uses MCP Code Q&A server to analyze Python repositories
and produce a detailed architecture and design report.

USAGE:
    python repo_analysis_agent.py path/to/repo --output report.md --mcp-server mcp_server.py
"""

import asyncio
import argparse
from pathlib import Path
from datetime import datetime
import sys
from typing import List, Dict, Any
import logging
import os

from analysis_reducer import AnalysisReducer
from mcp_client import MCPClient

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)


class RepositoryAnalysisAgent:
    def __init__(self, mcp_server_path: str):
        self.mcp_server_path = mcp_server_path
        self.mcp_client = MCPClient()
        self.analysis_reducer = AnalysisReducer()
        self.analysis_questions = {
            "architecture": [
                "What is the main entry point of this application?",
                "Describe the main modules and their responsibilities.",
            ],
            "dependencies": [
                "List all external Python packages used.",
                "What are the main import statements across the codebase?",
            ],
        }

    async def connect(self):
        await self.mcp_client.connect_to_server(self.mcp_server_path)

    async def close(self):
        await self.mcp_client.close()

    async def ask(self, question: str) -> str:
        answer = await self.mcp_client.process_query(question)
        return answer

    def static_file_scan(self, repo_path: str) -> Dict[str, Any]:
        """Collect static repo metrics."""
        repo = Path(repo_path)
        metrics = {
            "total_files": 0,
            "python_files": 0,
            "test_files": 0,
            "config_files": [],
            "doc_files": [],
            "directories": set(),
        }
        config_file_names = {
            "requirements.txt",
            "setup.py",
            "pyproject.toml",
            "setup.cfg",
            "Pipfile",
            "poetry.lock",
            "environment.yml",
            "Dockerfile",
            ".env",
        }
        doc_file_names = {
            "README.md",
            "README.rst",
            "README.txt",
            "CONTRIBUTING.md",
            "CHANGELOG.md",
            "LICENSE",
        }
        for item in repo.rglob("*"):
            if item.is_file():
                metrics["total_files"] += 1
                if item.suffix == ".py":
                    metrics["python_files"] += 1
                    if (
                        item.name.lower().startswith("test")
                        or "test" in item.name.lower()
                    ):
                        metrics["test_files"] += 1
                if item.name in config_file_names:
                    metrics["config_files"].append(item.name)
                if item.name in doc_file_names or item.suffix in {
                    ".md",
                    ".rst",
                    ".txt",
                }:
                    metrics["doc_files"].append(item.name)
            elif item.is_dir():
                metrics["directories"].add(item.relative_to(repo).as_posix())
        metrics["directories"] = list(metrics["directories"])
        metrics["config_files"] = list(set(metrics["config_files"]))
        metrics["doc_files"] = list(set(metrics["doc_files"]))
        return metrics

    def dependency_scan(self, repo_path: str) -> List[str]:
        """Scan for requirements from requirements.txt, setup.py, pyproject.toml."""
        deps = set()
        repo = Path(repo_path)
        req_file = repo / "requirements.txt"
        if req_file.exists():
            for line in req_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    deps.add(
                        line.split("==")[0]
                        .split(">=")[0]
                        .split("<=")[0]
                        .split(">")[0]
                        .split("<")[0]
                        .strip()
                    )
        setup_file = repo / "setup.py"
        if setup_file.exists():
            content = setup_file.read_text(encoding="utf-8")
            if "install_requires" in content:
                deps.add("setup.py:install_requires")
        pyproject_file = repo / "pyproject.toml"
        if pyproject_file.exists():
            deps.add("pyproject.toml")
        return sorted(list(deps))

    async def analyze(self, repo_path: str) -> Dict[str, Any]:
        await self.connect()
        try:
            await self.mcp_client.process_query(f"index the repo {repo_path}")
            static_metrics = self.static_file_scan(repo_path)
            static_dependencies = self.dependency_scan(repo_path)

            results = {}
            for section, questions in self.analysis_questions.items():
                results[section] = []
                for q in questions:
                    logging.info(f"=" * 100)
                    logging.info(f"Analyzing Question: {q}")
                    ans = await self.ask(q)
                    results[section].append({"question": q, "answer": ans})
                    logging.info(f"=" * 100)
                    await asyncio.sleep(0.3)

            return {
                "metrics": static_metrics,
                "static_dependencies": static_dependencies,
                "qa": results,
            }
        finally:
            await self.close()

    def write_report(self, analysis: Dict[str, Any], repo_path: str, output_path: str):
        dt = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            "# Repository Analysis Report",
            "",
            f"**Repository:** `{repo_path}`",
            f"**Date:** {dt}",
            "",
            "## Executive Summary",
            "",
            f"- Python files: **{analysis['metrics']['python_files']}**",
            f"- Test files: **{analysis['metrics']['test_files']}**",
            f"- Total files: **{analysis['metrics']['total_files']}**",
            f"- Config files: **{', '.join(analysis['metrics']['config_files']) or 'None'}**",
            f"- Documentation files: **{', '.join(analysis['metrics']['doc_files']) or 'None'}**",
            "",
            "## Directory Structure",
            "",
        ]
        for d in sorted(analysis["metrics"]["directories"])[:20]:
            lines.append(f"- `{d}`")
        lines.append("")
        lines.append("## Static Dependency Analysis\n")
        if analysis["static_dependencies"]:
            for dep in analysis["static_dependencies"]:
                lines.append(f"- `{dep}`")
        else:
            lines.append("No dependencies found in static files.")
        lines.append("")

        # Main report sections
        for section, qas in analysis["qa"].items():
            lines.append(f"## {section.capitalize()} Analysis\n")
            for qa in qas:
                lines.append(f"**Q:** {qa['question']}\n")
                ans = qa["answer"].strip()
                if ans:
                    lines.append(
                        f"**A:** {ans[:1600]}{'...' if len(ans) > 1600 else ''}\n"
                    )
                else:
                    lines.append("**A:** *(No answer)*\n")
            lines.append("")

        # Recommendations (simple heuristics)
        lines.append("## Recommendations\n")
        recs = []
        if analysis["metrics"]["test_files"] < max(
            1, analysis["metrics"]["python_files"] // 3
        ):
            recs.append("Consider improving test coverage.")
        if len(analysis["metrics"]["doc_files"]) < 2:
            recs.append(
                "Consider adding more documentation (README, contributing guide, etc)."
            )
        if not analysis["metrics"]["config_files"]:
            recs.append(
                "Consider adding configuration files for dependencies and environments."
            )
        if not recs:
            recs.append("No immediate issues detected. Codebase looks well structured.")
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")

        lines.append(
            "\n---\n*Report generated automatically by Repository Analysis Agent.*"
        )
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
        logging.info(f"Report written to {output_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Analyze a Python repository using MCP"
    )
    parser.add_argument("--repo-path", required=True, help="Path to the repository")
    parser.add_argument(
        "--mcp-server",
        default="mcp_server.py",
        help="Path to MCP server script",
    )
    args = parser.parse_args()
    if not os.path.exists(args.repo_path):
        print(f"Error: Repository path '{args.repo_path}' does not exist.")
        sys.exit(1)

    repo_path = args.repo_path
    agent = RepositoryAnalysisAgent(args.mcp_server)
    analysis = await agent.analyze(str(repo_path))
    await agent.analysis_reducer.write_report(
        analysis, str(repo_path), "analysis_report.md"
    )
    print(f"Analysis complete. See report at: analysis_report.md")


if __name__ == "__main__":
    asyncio.run(main())
