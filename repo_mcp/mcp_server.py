#!/usr/bin/env python3
"""
MCP Server for Code Q&A with RAG System
A Model Context Protocol server that answers questions about local code repositories
using Retrieval Augmented Generation (RAG).
"""

import asyncio
import os
from typing import List, Dict, Any, Optional
import logging
from rag_system import RAGSystem

# Third-party imports
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP Server Implementation
app = Server("code-qa-server")

# Global RAG system instance
rag_system: Optional[RAGSystem] = None


@app.list_tools()
async def list_tools() -> List[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="index_repository",
            description="Index a Python repository for Q&A",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to the Python repository to index",
                    }
                },
                "required": ["repo_path"],
            },
        ),
        types.Tool(
            name="ask_question",
            description="Ask a question about the indexed codebase",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question about the code",
                    }
                },
                "required": ["question"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    """Handle tool calls"""
    global rag_system

    if name == "index_repository":
        repo_path = arguments["repo_path"]

        if not os.path.exists(repo_path):
            return [
                types.TextContent(
                    type="text",
                    text=f"Error: Repository path '{repo_path}' does not exist.",
                )
            ]

        try:
            rag_system = RAGSystem(repo_path)
            rag_system.index_repository()

            return [
                types.TextContent(
                    type="text",
                    text=f"Successfully indexed repository at '{repo_path}'. ",
                )
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text", text=f"Error indexing repository: {str(e)}"
                )
            ]

    elif name == "ask_question":
        if rag_system is None:
            return [
                types.TextContent(
                    type="text",
                    text="Error: No repository has been indexed yet. Please index a repository first.",
                )
            ]

        question = arguments["question"]
        try:
            answer = rag_system.answer_question(question)
            return [types.TextContent(type="text", text=answer)]
        except Exception as e:
            return [
                types.TextContent(
                    type="text", text=f"Error answering question: {str(e)}"
                )
            ]
    else:
        return [
            types.TextContent(
                type="text",
                text="No such tool. Available tools: index_repository, ask_question",
            )
        ]


async def main():
    """Main entry point"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
