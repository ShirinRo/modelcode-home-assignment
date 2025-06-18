#!/usr/bin/env python3
"""
Analysis Reducer Class

This class provides LLM-powered reduction and summarization of repository analysis data
into professional markdown reports.

USAGE:
    from analysis_reducer import AnalysisReducer

    reducer = AnalysisReducer(api_key="your-api-key")

    # Reduce a single section
    section_md = await reducer.reduce_section("architecture", qa_data)

    # Reduce all sections into final report
    final_report = await reducer.reduce_sections(reduced_sections, analysis_data, repo_path)
"""

import asyncio
import logging
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, List, Any

import anthropic

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
)


class AnalysisReducer:
    """
    LLM-powered analysis reducer that converts Q&A data into professional markdown reports.
    """

    def __init__(self):
        """
        Initialize the Analysis Reducer.

        Args:
            api_key: Anthropic API key
            model: Claude model to use for text generation
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key is required for Analysis Reducer")

        self.model = "claude-3-5-sonnet-20241022"
        self.client = anthropic.Anthropic(api_key=api_key)

    async def _call_llm(self, prompt: str, max_tokens: int = 4000) -> str:
        """
        Make an API call to Claude using the Anthropic package.

        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens in the response

        Returns:
            The LLM's response text
        """
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            content = message.content[0]
            if hasattr(content, "text") and hasattr(content, "type"):
                if content.type == "text":
                    return content.text
            return ""
        except Exception as e:
            logging.error(f"LLM API call failed: {e}")
            raise

    async def _reduce_section(
        self, section_name: str, qa_data: List[Dict[str, str]]
    ) -> str:
        """
        Reduce a single analysis section's Q&A data into a comprehensive professional markdown.

        Args:
            section_name: Name of the section (e.g., 'architecture', 'dependencies')
            qa_data: List of dictionaries with 'question' and 'answer' keys

        Returns:
            Professional markdown content for the section
        """
        logging.info(f"Reducing section: {section_name}")

        # Prepare the Q&A content for the LLM
        qa_content = ""
        for i, qa in enumerate(qa_data, 1):
            qa_content += f"**Question {i}:** {qa['question']}\n"
            qa_content += f"**Answer {i}:** {qa['answer']}\n\n"

        section_prompts = {
            "architecture": "Focus on system design, entry points, module organization, class relationships, architectural patterns, and separation of concerns.",
            "dependencies": "Focus on external packages, imports, third-party integrations, database connections, configuration files, and dependency management.",
            "patterns": "Focus on design patterns, decorators, context managers, error handling strategies, and configuration management approaches.",
            "components": "Focus on key system components, inter-module communication, data structures, APIs, CLI interfaces, testing frameworks, and monitoring.",
            "quality": "Focus on documentation quality, test coverage, code maintainability, technical debt, and areas for improvement.",
        }

        specific_guidance = section_prompts.get(
            section_name,
            "Focus on the key technical aspects and insights from this analysis section.",
        )

        prompt = f"""You are an expert technical writer and software architect. You have been given Q&A data from a repository analysis focused on the {section_name} aspects of a codebase.

Your task is to create a comprehensive, well-structured markdown section that synthesizes all the information from the Q&A pairs below into a professional technical report.

**Section Focus:** {specific_guidance}

**Requirements:**
1. Create clear, professional markdown with appropriate headers (##, ###)
2. Synthesize information rather than just listing Q&As
3. Identify patterns, relationships, and key technical insights
4. Use bullet points, code blocks, and formatting appropriately
5. Highlight important findings, architectural decisions, and technical details
6. Structure content logically with subheadings
7. Make it readable for both technical and non-technical stakeholders
8. Focus on actionable insights and clear explanations
9. Include specific examples and technical details when available
10. Avoid redundancy and ensure smooth flow

**Q&A Data to synthesize:**
{qa_content}

Please create a comprehensive markdown section for **{section_name.upper()}** that synthesizes this information professionally. Start with a brief overview, then organize the content into logical subsections with clear headers:"""

        try:
            response = await self._call_llm(prompt, max_tokens=6000)
            logging.info(f"Successfully reduced section: {section_name}")
            return response
        except Exception as e:
            logging.error(f"Failed to reduce section {section_name}: {e}")
            return self._create_fallback_section_summary(section_name, qa_data)

    async def _reduce_sections(
        self,
        reduced_sections: Dict[str, str],
        analysis_data: Dict[str, Any],
        repo_path: str,
    ) -> str:
        """
        Reduce all section summaries and analysis data into a comprehensive repository report.

        Args:
            reduced_sections: Dictionary mapping section names to their reduced markdown content
            analysis_data: Complete analysis data including metrics, dependencies, and Q&A
            repo_path: Path to the analyzed repository

        Returns:
            Complete professional markdown report
        """
        logging.info("Creating comprehensive repository report")

        # Extract metrics for statistics section
        metrics = analysis_data.get("metrics", {})
        static_deps = analysis_data.get("static_dependencies", [])

        # Create statistics section content
        stats_content = f"""
**Repository Metrics:**
- **Python files:** {metrics.get('python_files', 0)}
- **Test files:** {metrics.get('test_files', 0)}
- **Total files:** {metrics.get('total_files', 0)}
- **Configuration files:** {', '.join(metrics.get('config_files', [])) if metrics.get('config_files') else 'None'}
- **Documentation files:** {', '.join(metrics.get('doc_files', [])) if metrics.get('doc_files') else 'None'}

**Directory Structure (Top Level):**
{chr(10).join(f'- `{d}`' for d in sorted(metrics.get('directories', []))[:15])}
{f'- ... and {len(metrics.get("directories", [])) - 15} more directories' if len(metrics.get('directories', [])) > 15 else ''}

**Static Dependencies:**
{chr(10).join(f'- `{dep}`' for dep in static_deps) if static_deps else '- No dependencies found in static files'}
"""

        # Prepare sections content
        sections_content = ""
        for section_name, content in reduced_sections.items():
            sections_content += (
                f"## {section_name.capitalize()} Analysis\n\n{content}\n\n"
            )

        prompt = f"""You are an expert technical writer and software architect. You have detailed analysis sections for a repository and need to create a comprehensive, executive-level summary report.

**Repository:** `{repo_path}`
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

**Repository Statistics:**
{stats_content}

**Detailed Analysis Sections:**
{sections_content}

Your task is to create a comprehensive repository analysis report in markdown format with the following structure:

1. **Title and Overview** - Professional title and brief repository description
2. **Executive Summary** - 2-3 paragraph high-level overview with key findings
3. **Repository Statistics** - Include the provided statistics in a well-formatted section
4. **Technical Analysis** - Incorporate the provided analysis sections, ensuring they flow well together
5. **Key Insights** - Synthesize the most important findings across all analysis areas
6. **Technical Recommendations** - Specific, actionable recommendations for improvement
7. **Conclusion** - Brief summary of overall assessment

**Requirements:**
- Professional, executive-ready report suitable for technical leads and stakeholders
- Clear structure with consistent markdown formatting
- Highlight critical findings, architectural strengths, and areas for improvement
- Provide specific, actionable recommendations with priorities
- Balance technical depth with accessibility
- Use tables, bullet points, and formatting effectively
- Ensure logical flow between sections
- Include a brief table of contents after the title
- Focus on insights that drive decision-making

Create a comprehensive, professional repository analysis report:"""

        try:
            response = await self._call_llm(prompt, max_tokens=8000)
            logging.info("Successfully created comprehensive report")
            return response
        except Exception as e:
            logging.error(f"Failed to create comprehensive report: {e}")
            return self._create_fallback_comprehensive_report(
                reduced_sections, analysis_data, repo_path
            )

    def _create_fallback_section_summary(
        self, section_name: str, qa_data: List[Dict[str, str]]
    ) -> str:
        """Create a basic fallback summary if LLM call fails for a section."""
        content = [f"## {section_name.capitalize()} Analysis", ""]

        for i, qa in enumerate(qa_data, 1):
            content.extend(
                [
                    f"### Question {i}: {qa['question']}",
                    "",
                    qa["answer"][:1000] + ("..." if len(qa["answer"]) > 1000 else ""),
                    "",
                ]
            )

        content.extend(
            ["---", "*Note: This is a fallback summary due to processing limitations.*"]
        )

        return "\n".join(content)

    def _create_fallback_comprehensive_report(
        self,
        reduced_sections: Dict[str, str],
        analysis_data: Dict[str, Any],
        repo_path: str,
    ) -> str:
        """Create a fallback comprehensive report if LLM call fails."""
        metrics = analysis_data.get("metrics", {})
        static_deps = analysis_data.get("static_dependencies", [])

        lines = [
            "# Repository Analysis Report",
            "",
            f"**Repository:** `{repo_path}`",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Analysis Type:** Comprehensive Technical Analysis",
            "",
            "## Executive Summary",
            "",
            f"This repository contains {metrics.get('python_files', 0)} Python files across {len(metrics.get('directories', []))} directories. "
            f"The analysis covers architecture, dependencies, design patterns, components, and code quality.",
            "",
            "## Repository Statistics",
            "",
            f"- **Python files:** {metrics.get('python_files', 0)}",
            f"- **Test files:** {metrics.get('test_files', 0)}",
            f"- **Total files:** {metrics.get('total_files', 0)}",
            f"- **Configuration files:** {', '.join(metrics.get('config_files', [])) if metrics.get('config_files') else 'None'}",
            f"- **Documentation files:** {', '.join(metrics.get('doc_files', [])) if metrics.get('doc_files') else 'None'}",
            "",
            "### Directory Structure",
            "",
        ]

        for d in sorted(metrics.get("directories", []))[:20]:
            lines.append(f"- `{d}`")

        lines.extend(
            [
                "",
                "### Static Dependencies",
                "",
            ]
        )

        if static_deps:
            for dep in static_deps:
                lines.append(f"- `{dep}`")
        else:
            lines.append("- No dependencies found in static files")

        lines.extend(["", "## Technical Analysis", ""])

        # Add all reduced sections
        for section_name, content in reduced_sections.items():
            lines.extend(
                [
                    f"### {section_name.capitalize()}",
                    "",
                    content,
                    "",
                ]
            )

        # Add basic recommendations
        lines.extend(
            [
                "## Recommendations",
                "",
                "Based on the analysis, consider the following improvements:",
                "",
            ]
        )

        recs = []
        if metrics.get("test_files", 0) < max(1, metrics.get("python_files", 0) // 3):
            recs.append(
                "**Improve test coverage** - Current test-to-code ratio suggests insufficient testing"
            )

        if len(metrics.get("doc_files", [])) < 2:
            recs.append(
                "**Enhance documentation** - Add comprehensive README, API docs, and development guides"
            )

        if not metrics.get("config_files"):
            recs.append(
                "**Add configuration management** - Implement proper dependency and environment configuration"
            )

        if not recs:
            recs.append(
                "**Maintain current quality** - No critical issues identified in static analysis"
            )

        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")

        lines.extend(
            [
                "",
                "---",
                "*Report generated by Analysis Reducer with fallback processing.*",
            ]
        )

        return "\n".join(lines)

    async def write_report(
        self, analysis: Dict[str, Any], repo_path: str, output_report_filename: str
    ):
        """
        Enhanced write_report function that uses LLM reduction to create a cohesive markdown report.

        Args:
            analysis: Complete analysis data including metrics, dependencies, and Q&A
            repo_path: Path to the analyzed repository
            output_path: Path where the final report should be written
        """
        logging.info(f"Creating enhanced report for {repo_path}")
        results_directory = "analyzer/results"
        Path(results_directory).mkdir(parents=True, exist_ok=True)
        sections_results_directory = results_directory + "/sections"
        Path(sections_results_directory).mkdir(parents=True, exist_ok=True)
        output_path = f"{results_directory}/{output_report_filename}"

        try:
            # Step 1: Reduce each section using LLM
            logging.info("Reducing individual sections...")
            reduced_sections = {}

            for section_name, qa_data in analysis["qa"].items():
                logging.info(f"Reducing {section_name} section...")
                reduced_content = await self._reduce_section(section_name, qa_data)
                reduced_sections[section_name] = reduced_content

                # Optional: Save individual section files for reference
                section_file = f"{sections_results_directory}/section_{section_name}.md"
                Path(section_file).write_text(reduced_content, encoding="utf-8")
                logging.info(f"Section {section_name} saved to {section_file}")

                # Rate limiting between API calls
                await asyncio.sleep(0.5)

            # Step 2: Merge all sections with statistics into comprehensive report
            logging.info("Creating comprehensive merged report...")
            comprehensive_report = await self._reduce_sections(
                reduced_sections, analysis, repo_path
            )

            # Step 3: Write the final report to disk
            Path(output_path).write_text(comprehensive_report, encoding="utf-8")
            logging.info(f"Enhanced comprehensive report written to {output_path}")

        except Exception as e:
            logging.error(f"Enhanced report generation failed: {e}")
            logging.info("Falling back to basic report format...")

            # Fallback: Create basic report if LLM enhancement fails
            self._write_basic_fallback_report(analysis, repo_path, output_path)

    def _write_basic_fallback_report(
        self, analysis: Dict[str, Any], repo_path: str, output_path: str
    ):
        """Fallback method to write basic report if LLM enhancement fails."""
        dt = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            "# Repository Analysis Report (Fallback)",
            "",
            f"**Repository:** `{repo_path}`",
            f"**Date:** {dt}",
            f"**Note:** Enhanced analysis failed, showing basic report.",
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

        # Main report sections (basic Q&A format)
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

        lines.append("\n---\n*Fallback report generated by Repository Analysis Agent.*")
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
        logging.info(f"Fallback report written to {output_path}")
