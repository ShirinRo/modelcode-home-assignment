#!/usr/bin/env python3
"""
Repository Analysis Agent
Uses the MCP Code QA server to analyze a repository and generate comprehensive reports
about architecture, dependencies, and design patterns.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass, asdict
import subprocess
import re
from datetime import datetime

# Import our systems
from mcp_code_qa_server import RAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ArchitectureInfo:
    """Information about repository architecture"""

    main_modules: List[str]
    package_structure: Dict[str, List[str]]
    entry_points: List[str]
    key_classes: List[Dict[str, Any]]
    key_functions: List[Dict[str, Any]]
    design_patterns: List[str]


@dataclass
class DependencyInfo:
    """Information about external dependencies"""

    external_imports: List[str]
    requirements: List[str]
    dependency_categories: Dict[str, List[str]]
    potential_issues: List[str]


@dataclass
class AnalysisReport:
    """Complete repository analysis report"""

    repository_path: str
    analysis_date: str
    summary: str
    architecture: ArchitectureInfo
    dependencies: DependencyInfo
    code_quality: Dict[str, Any]
    recommendations: List[str]


class RepoAnalyzer:
    """Analyzes repository structure and patterns"""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.rag_system = RAGSystem(repo_path)

    def analyze_repository(self) -> AnalysisReport:
        """Perform comprehensive repository analysis"""
        logger.info(f"Starting analysis of repository: {self.repo_path}")

        # Index the repository first
        self.rag_system.index_repository()

        # Gather information using various analysis methods
        architecture = self._analyze_architecture()
        dependencies = self._analyze_dependencies()
        code_quality = self._analyze_code_quality()

        # Generate summary and recommendations
        summary = self._generate_summary(architecture, dependencies, code_quality)
        recommendations = self._generate_recommendations(
            architecture, dependencies, code_quality
        )

        return AnalysisReport(
            repository_path=str(self.repo_path),
            analysis_date=datetime.now().isoformat(),
            summary=summary,
            architecture=architecture,
            dependencies=dependencies,
            code_quality=code_quality,
            recommendations=recommendations,
        )

    def _analyze_architecture(self) -> ArchitectureInfo:
        """Analyze repository architecture"""
        logger.info("Analyzing architecture...")

        # Find main modules and packages
        main_modules = []
        package_structure = {}

        for py_file in self.repo_path.rglob("*.py"):
            relative_path = py_file.relative_to(self.repo_path)
            parts = relative_path.parts

            # Identify main modules (top-level .py files or __main__.py)
            if len(parts) == 1 or parts[-1] == "__main__.py":
                main_modules.append(str(relative_path))

            # Build package structure
            if len(parts) > 1:
                package = parts[0]
                if package not in package_structure:
                    package_structure[package] = []
                package_structure[package].append(str(relative_path))

        # Find entry points
        entry_points = self._find_entry_points()

        # Extract key classes and functions using RAG system
        key_classes = self._extract_key_classes()
        key_functions = self._extract_key_functions()

        # Identify design patterns
        design_patterns = self._identify_design_patterns()

        return ArchitectureInfo(
            main_modules=main_modules,
            package_structure=package_structure,
            entry_points=entry_points,
            key_classes=key_classes,
            key_functions=key_functions,
            design_patterns=design_patterns,
        )

    def _analyze_dependencies(self) -> DependencyInfo:
        """Analyze external dependencies"""
        logger.info("Analyzing dependencies...")

        # Extract imports from code chunks
        external_imports = set()
        for chunk in self.rag_system.chunks:
            if chunk.chunk_type == "import":
                imports = self._parse_imports(chunk.content)
                external_imports.update(imports)

        # Read requirements files
        requirements = self._read_requirements()

        # Categorize dependencies
        dependency_categories = self._categorize_dependencies(
            list(external_imports), requirements
        )

        # Identify potential issues
        potential_issues = self._identify_dependency_issues(
            list(external_imports), requirements
        )

        return DependencyInfo(
            external_imports=sorted(list(external_imports)),
            requirements=requirements,
            dependency_categories=dependency_categories,
            potential_issues=potential_issues,
        )

    def _analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze code quality metrics"""
        logger.info("Analyzing code quality...")

        total_chunks = len(self.rag_system.chunks)
        chunk_types = {}

        # Count chunk types
        for chunk in self.rag_system.chunks:
            chunk_type = chunk.chunk_type
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

        # Calculate metrics
        classes_with_docstrings = sum(
            1 for c in self.rag_system.chunks if c.chunk_type == "class" and c.docstring
        )
        functions_with_docstrings = sum(
            1
            for c in self.rag_system.chunks
            if c.chunk_type in ["function", "method"] and c.docstring
        )

        total_classes = chunk_types.get("class", 0)
        total_functions = chunk_types.get("function", 0) + chunk_types.get("method", 0)

        documentation_coverage = {
            "classes": classes_with_docstrings / max(total_classes, 1) * 100,
            "functions": functions_with_docstrings / max(total_functions, 1) * 100,
        }

        # Complexity estimation (based on dependencies)
        avg_dependencies = sum(
            len(c.dependencies) for c in self.rag_system.chunks
        ) / max(total_chunks, 1)

        return {
            "total_chunks": total_chunks,
            "chunk_distribution": chunk_types,
            "documentation_coverage": documentation_coverage,
            "average_dependencies_per_chunk": avg_dependencies,
            "lines_of_code": self._estimate_loc(),
        }

    def _find_entry_points(self) -> List[str]:
        """Find potential entry points in the repository"""
        entry_points = []

        # Look for common entry point patterns
        patterns = ["main.py", "__main__.py", "app.py", "run.py", "server.py", "cli.py"]

        for pattern in patterns:
            matches = list(self.repo_path.rglob(pattern))
            entry_points.extend([str(m.relative_to(self.repo_path)) for m in matches])

        # Look for if __name__ == "__main__" patterns
        for py_file in self.repo_path.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if 'if __name__ == "__main__"' in content:
                        entry_points.append(str(py_file.relative_to(self.repo_path)))
            except Exception:
                continue

        return list(set(entry_points))  # Remove duplicates

    def _extract_key_classes(self) -> List[Dict[str, Any]]:
        """Extract information about key classes"""
        class_chunks = [c for c in self.rag_system.chunks if c.chunk_type == "class"]

        key_classes = []
        for chunk in class_chunks[:10]:  # Top 10 classes
            key_classes.append(
                {
                    "name": chunk.name,
                    "file": chunk.file_path,
                    "line": chunk.start_line,
                    "has_docstring": bool(chunk.docstring),
                    "dependencies": chunk.dependencies[:5],  # Top 5 dependencies
                    "methods_count": len(
                        [
                            c
                            for c in self.rag_system.chunks
                            if c.chunk_type == "method" and c.parent_class == chunk.name
                        ]
                    ),
                }
            )

        return key_classes

    def _extract_key_functions(self) -> List[Dict[str, Any]]:
        """Extract information about key functions"""
        function_chunks = [
            c for c in self.rag_system.chunks if c.chunk_type == "function"
        ]

        key_functions = []
        for chunk in function_chunks[:10]:  # Top 10 functions
            key_functions.append(
                {
                    "name": chunk.name,
                    "file": chunk.file_path,
                    "line": chunk.start_line,
                    "has_docstring": bool(chunk.docstring),
                    "dependencies": chunk.dependencies[:5],  # Top 5 dependencies
                }
            )

        return key_functions

    def _identify_design_patterns(self) -> List[str]:
        """Identify common design patterns in the code"""
        patterns = []

        # Look for common patterns by analyzing class and function names
        class_names = [
            c.name.lower() for c in self.rag_system.chunks if c.chunk_type == "class"
        ]
        function_names = [
            c.name.lower() for c in self.rag_system.chunks if c.chunk_type == "function"
        ]

        # Singleton pattern
        if any("singleton" in name for name in class_names):
            patterns.append("Singleton Pattern")

        # Factory pattern
        if any("factory" in name for name in class_names + function_names):
            patterns.append("Factory Pattern")

        # Observer pattern
        if any(
            word in " ".join(class_names)
            for word in ["observer", "listener", "subscriber"]
        ):
            patterns.append("Observer Pattern")

        # Builder pattern
        if any("builder" in name for name in class_names):
            patterns.append("Builder Pattern")

        # MVC pattern
        mvc_indicators = ["controller", "model", "view"]
        if (
            sum(
                any(indicator in name for name in class_names)
                for indicator in mvc_indicators
            )
            >= 2
        ):
            patterns.append("MVC Pattern")

        # Command pattern
        if any("command" in name for name in class_names):
            patterns.append("Command Pattern")

        # Strategy pattern
        if any("strategy" in name for name in class_names):
            patterns.append("Strategy Pattern")

        return patterns

    def _parse_imports(self, import_content: str) -> List[str]:
        """Parse import statements to extract external dependencies"""
        imports = []
        lines = import_content.split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("import "):
                module = line.replace("import ", "").split(".")[0].split(" as ")[0]
                imports.append(module)
            elif line.startswith("from "):
                match = re.match(r"from\s+(\w+)", line)
                if match:
                    imports.append(match.group(1))

        # Filter out standard library modules (basic filtering)
        stdlib_modules = {
            "os",
            "sys",
            "json",
            "time",
            "datetime",
            "re",
            "math",
            "random",
            "collections",
            "itertools",
            "functools",
            "typing",
            "pathlib",
            "logging",
            "unittest",
            "asyncio",
            "threading",
            "multiprocessing",
        }

        return [imp for imp in imports if imp not in stdlib_modules]

    def _read_requirements(self) -> List[str]:
        """Read requirements from requirements files"""
        requirements = []

        # Common requirements file names
        req_files = [
            "requirements.txt",
            "requirements.in",
            "setup.py",
            "pyproject.toml",
            "Pipfile",
        ]

        for req_file in req_files:
            file_path = self.repo_path / req_file
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    if req_file == "requirements.txt" or req_file == "requirements.in":
                        reqs = [
                            line.strip().split("==")[0].split(">=")[0].split("<=")[0]
                            for line in content.split("\n")
                            if line.strip() and not line.startswith("#")
                        ]
                        requirements.extend(reqs)
                    elif req_file == "setup.py":
                        # Basic parsing of setup.py install_requires
                        matches = re.findall(
                            r"install_requires\s*=\s*\[(.*?)\]", content, re.DOTALL
                        )
                        if matches:
                            req_text = matches[0]
                            reqs = re.findall(r'["\']([^"\']+)["\']', req_text)
                            requirements.extend(
                                [req.split("==")[0].split(">=")[0] for req in reqs]
                            )

                except Exception as e:
                    logger.warning(f"Error reading {req_file}: {e}")

        return list(set(requirements))  # Remove duplicates

    def _categorize_dependencies(
        self, imports: List[str], requirements: List[str]
    ) -> Dict[str, List[str]]:
        """Categorize dependencies by type"""
        categories = {
            "web_frameworks": [],
            "data_science": [],
            "testing": [],
            "development": [],
            "database": [],
            "other": [],
        }

        all_deps = list(set(imports + requirements))

        # Categorization rules
        web_frameworks = {"flask", "django", "fastapi", "tornado", "pyramid", "bottle"}
        data_science = {
            "numpy",
            "pandas",
            "scipy",
            "matplotlib",
            "seaborn",
            "plotly",
            "sklearn",
            "tensorflow",
            "pytorch",
        }
        testing = {"pytest", "unittest", "nose", "mock", "coverage"}
        development = {"black", "flake8", "mypy", "pylint", "pre-commit"}
        database = {"sqlalchemy", "pymongo", "psycopg2", "mysql", "sqlite3"}

        for dep in all_deps:
            dep_lower = dep.lower()
            if any(fw in dep_lower for fw in web_frameworks):
                categories["web_frameworks"].append(dep)
            elif any(ds in dep_lower for ds in data_science):
                categories["data_science"].append(dep)
            elif any(test in dep_lower for test in testing):
                categories["testing"].append(dep)
            elif any(dev in dep_lower for dev in development):
                categories["development"].append(dep)
            elif any(db in dep_lower for db in database):
                categories["database"].append(dep)
            else:
                categories["other"].append(dep)

        return categories

    def _identify_dependency_issues(
        self, imports: List[str], requirements: List[str]
    ) -> List[str]:
        """Identify potential dependency issues"""
        issues = []

        # Missing in requirements
        missing_in_req = set(imports) - set(requirements)
        if missing_in_req:
            issues.append(
                f"Imports not in requirements: {', '.join(list(missing_in_req)[:5])}"
            )

        # Unused requirements
        unused_req = set(requirements) - set(imports)
        if unused_req:
            issues.append(f"Unused requirements: {', '.join(list(unused_req)[:5])}")

        return issues

    def _estimate_loc(self) -> int:
        """Estimate lines of code"""
        total_lines = 0
        for py_file in self.repo_path.rglob("*.py"):
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    total_lines += len(f.readlines())
            except Exception:
                continue
        return total_lines

    def _generate_summary(
        self,
        architecture: ArchitectureInfo,
        dependencies: DependencyInfo,
        code_quality: Dict[str, Any],
    ) -> str:
        """Generate a summary of the repository"""
        summary_parts = []

        # Basic info
        summary_parts.append(
            f"This repository contains {code_quality['total_chunks']} code chunks across {len(architecture.main_modules)} main modules."
        )

        # Architecture
        if architecture.key_classes:
            summary_parts.append(
                f"The codebase defines {len(architecture.key_classes)} key classes and {len(architecture.key_functions)} main functions."
            )

        # Dependencies
        if dependencies.external_imports:
            summary_parts.append(
                f"It uses {len(dependencies.external_imports)} external libraries including {', '.join(dependencies.external_imports[:3])}."
            )

        # Patterns
        if architecture.design_patterns:
            summary_parts.append(
                f"The code implements several design patterns: {', '.join(architecture.design_patterns)}."
            )

        # Quality
        doc_coverage = code_quality["documentation_coverage"]
        summary_parts.append(
            f"Documentation coverage is {doc_coverage['classes']:.1f}% for classes and {doc_coverage['functions']:.1f}% for functions."
        )

        return " ".join(summary_parts)

    def _generate_recommendations(
        self,
        architecture: ArchitectureInfo,
        dependencies: DependencyInfo,
        code_quality: Dict[str, Any],
    ) -> List[str]:
        """Generate recommendations for the repository"""
        recommendations = []

        # Documentation recommendations
        doc_coverage = code_quality["documentation_coverage"]
        if doc_coverage["classes"] < 70:
            recommendations.append(
                "Consider improving class documentation - only {:.1f}% of classes have docstrings".format(
                    doc_coverage["classes"]
                )
            )

        if doc_coverage["functions"] < 70:
            recommendations.append(
                "Consider improving function documentation - only {:.1f}% of functions have docstrings".format(
                    doc_coverage["functions"]
                )
            )

        # Dependency recommendations
        if dependencies.potential_issues:
            recommendations.extend(
                [
                    f"Dependency issue: {issue}"
                    for issue in dependencies.potential_issues
                ]
            )

        # Architecture recommendations
        if not architecture.entry_points:
            recommendations.append(
                "No clear entry points found - consider adding a main.py or __main__.py file"
            )

        if code_quality["average_dependencies_per_chunk"] > 10:
            recommendations.append(
                "High coupling detected - consider reducing dependencies between modules"
            )

        # Pattern recommendations
        if not architecture.design_patterns:
            recommendations.append(
                "No clear design patterns detected - consider implementing common patterns for better maintainability"
            )

        return recommendations


class ReportGenerator:
    """Generates formatted reports from analysis results"""

    @staticmethod
    def generate_markdown_report(
        analysis: AnalysisReport, output_file: str = "repository_analysis.md"
    ):
        """Generate a markdown report"""

        report_content = f"""# Repository Analysis Report

**Repository:** {analysis.repository_path}  
**Analysis Date:** {analysis.analysis_date}

## Executive Summary

{analysis.summary}

## Architecture Overview

### Main Modules
{chr(10).join([f"- {module}" for module in analysis.architecture.main_modules])}

### Package Structure
"""

        for package, files in analysis.architecture.package_structure.items():
            report_content += f"- **{package}/** ({len(files)} files)\n"

        report_content += f"""
### Entry Points
{chr(10).join([f"- {entry}" for entry in analysis.architecture.entry_points])}

### Key Classes
| Name | File | Methods | Documentation |
|------|------|---------|---------------|
"""

        for cls in analysis.architecture.key_classes:
            doc_status = "✅" if cls["has_docstring"] else "❌"
            report_content += f"| {cls['name']} | {Path(cls['file']).name} | {cls['methods_count']} | {doc_status} |\n"

        report_content += f"""
### Key Functions
| Name | File | Documentation |
|------|------|---------------|
"""

        for func in analysis.architecture.key_functions:
            doc_status = "✅" if func["has_docstring"] else "❌"
            report_content += (
                f"| {func['name']} | {Path(func['file']).name} | {doc_status} |\n"
            )

        report_content += f"""
### Design Patterns
{chr(10).join([f"- {pattern}" for pattern in analysis.architecture.design_patterns]) if analysis.architecture.design_patterns else "No clear design patterns detected."}

## Dependencies

### External Dependencies
{chr(10).join([f"- {dep}" for dep in analysis.dependencies.external_imports[:20]])}

### Dependency Categories
"""

        for category, deps in analysis.dependencies.dependency_categories.items():
            if deps:
                report_content += (
                    f"- **{category.replace('_', ' ').title()}:** {', '.join(deps)}\n"
                )

        if analysis.dependencies.potential_issues:
            report_content += f"""
### Potential Issues
{chr(10).join([f"- {issue}" for issue in analysis.dependencies.potential_issues])}
"""

        report_content += f"""
## Code Quality Metrics

- **Total Code Chunks:** {analysis.code_quality['total_chunks']}
- **Lines of Code:** {analysis.code_quality['lines_of_code']:,}
- **Class Documentation Coverage:** {analysis.code_quality['documentation_coverage']['classes']:.1f}%
- **Function Documentation Coverage:** {analysis.code_quality['documentation_coverage']['functions']:.1f}%
- **Average Dependencies per Chunk:** {analysis.code_quality['average_dependencies_per_chunk']:.1f}

### Chunk Distribution
"""

        for chunk_type, count in analysis.code_quality["chunk_distribution"].items():
            report_content += f"- **{chunk_type.title()}:** {count}\n"

        report_content += f"""
## Recommendations

{chr(10).join([f"- {rec}" for rec in analysis.recommendations])}

---
*Report generated by Repository Analysis Agent*
"""

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        return report_content

    @staticmethod
    def generate_json_report(
        analysis: AnalysisReport, output_file: str = "repository_analysis.json"
    ):
        """Generate a JSON report"""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(asdict(analysis), f, indent=2, default=str)


def main():
    """Main function for repository analysis"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze Repository Architecture and Dependencies"
    )
    parser.add_argument("repo_path", help="Path to the repository to analyze")
    parser.add_argument(
        "--output-format",
        choices=["markdown", "json", "both"],
        default="both",
        help="Output format for the report",
    )
    parser.add_argument("--output-dir", default=".", help="Directory to save reports")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate repository path
    if not os.path.exists(args.repo_path):
        print(f"Error: Repository path '{args.repo_path}' does not exist.")
        sys.exit(1)

    print(f"Starting repository analysis: {args.repo_path}")

    # Initialize analyzer
    analyzer = RepoAnalyzer(args.repo_path)

    # Perform analysis
    print("Analyzing repository...")
    analysis = analyzer.analyze_repository()

    # Generate reports
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.output_format in ["markdown", "both"]:
        md_file = output_dir / "repository_analysis.md"
        ReportGenerator.generate_markdown_report(analysis, str(md_file))
        print(f"Markdown report saved to: {md_file}")

    if args.output_format in ["json", "both"]:
        json_file = output_dir / "repository_analysis.json"
        ReportGenerator.generate_json_report(analysis, str(json_file))
        print(f"JSON report saved to: {json_file}")

    # Print summary
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print(analysis.summary)

    if analysis.recommendations:
        print(f"\nKey Recommendations:")
        for i, rec in enumerate(analysis.recommendations[:5], 1):
            print(f"{i}. {rec}")


if __name__ == "__main__":
    main()
