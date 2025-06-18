import asyncio
from typing import List, Optional, cast
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from anthropic.types.tool_union_param import ToolUnionParam
from anthropic.types import (
    MessageParam,
)

from dotenv import load_dotenv

load_dotenv()


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack: Optional[AsyncExitStack] = None
        self.anthropic = Anthropic()
        self.conversation_history: List[MessageParam] = []
        self.stdio = None
        self.write = None

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server."""
        # Clean up any existing connection first
        if self.session is not None:
            await self.close()

        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        # Create new exit stack for this connection
        self.exit_stack = AsyncExitStack()

        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport

            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            await self.session.initialize()

            # Print available tools
            response = await self.session.list_tools()
            tools = response.tools
            print("\nConnected to server with tools:", [tool.name for tool in tools])

        except Exception as e:
            # If connection fails, clean up the exit stack
            if self.exit_stack:
                try:
                    await self.exit_stack.aclose()
                except Exception:
                    pass  # Ignore cleanup errors
                self.exit_stack = None
            raise e

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        if self.session is None:
            raise RuntimeError(
                "MCPClient not connected: call 'await connect_to_server(...)' first."
            )

        # Add new query to conversation history
        self.conversation_history.append({"role": "user", "content": query})

        response = await self.session.list_tools()
        available_tools: List[ToolUnionParam] = []
        for tool in response.tools:
            available_tools.append(
                cast(
                    ToolUnionParam,
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    },
                )
            )

        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=self.conversation_history,
            tools=available_tools,
        )

        tool_results = []
        final_text = []
        assistant_content = []

        for content in response.content:
            if content.type == "text":
                assistant_content.append({"type": "text", "text": content.text})
            elif content.type == "tool_use":
                tool_name = content.name
                tool_args = content.input

                # Add tool use to assistant content
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": content.id,
                        "name": tool_name,
                        "input": tool_args,
                    }
                )

                # Ensure tool_args is a dictionary
                if not isinstance(tool_args, dict):
                    raise TypeError(
                        f"Tool arguments for '{tool_name}' must be a dict, got {type(tool_args)}"
                    )

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                # Add assistant message with tool use
                self.conversation_history.append(
                    {"role": "assistant", "content": assistant_content}
                )

                # Add tool result as user message
                tool_result_text = self.extract_text_from_content(result.content)
                self.conversation_history.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": tool_result_text,
                            }
                        ],
                    }
                )

                # Get next response from Claude with updated conversation
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=self.conversation_history,
                    tools=available_tools,
                )

                # Add the final response text
                for final_content in response.content:
                    if final_content.type == "text":
                        final_text.append(final_content.text)

        # Add final assistant response to conversation history
        if response.content:
            final_assistant_content = []
            for content in response.content:
                if content.type == "text":
                    final_assistant_content.append(
                        {"type": "text", "text": content.text}
                    )

            if final_assistant_content:
                self.conversation_history.append(
                    {"role": "assistant", "content": final_assistant_content}
                )

        return "\n".join(final_text)

    def extract_text_from_content(self, content) -> str:
        """
        Extract text from MCP tool result content, handling various content types.
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if hasattr(item, "text") and hasattr(item, "type"):
                    if item.type == "text":
                        text_parts.append(item.text)
                    elif item.type == "image":
                        text_parts.append("[Image content]")
                    elif item.type == "audio":
                        text_parts.append("[Audio content]")
                    elif item.type == "resource":
                        if hasattr(item, "text"):
                            text_parts.append(item.text)
                        else:
                            text_parts.append(
                                f"[Resource: {getattr(item, 'uri', 'unknown')}]"
                            )
                    else:
                        text_parts.append(f"[{item.type} content]")
                elif hasattr(item, "text"):
                    text_parts.append(item.text)
                else:
                    text_parts.append(str(item))

            return "\n".join(text_parts) if text_parts else ""

        return str(content)

    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []

    async def close(self):
        """Properly close the MCP client connection"""
        if self.exit_stack is not None:
            try:
                await self.exit_stack.aclose()
            except Exception as e:
                # Log the error but don't raise it during cleanup
                print(f"Warning: Error during MCP client cleanup: {e}")
            finally:
                self.exit_stack = None
                self.session = None
                self.stdio = None
                self.write = None


async def main():
    client = MCPClient()
    await client.connect_to_server("mcp_server.py")

    # First query - index repo
    result1 = await client.process_query("index the repo grip-no-tests")
    print("First query result:", result1)

    # Second query - now Claude knows the repo is indexed
    result2 = await client.process_query("How do I run grip from command line?")
    print("Second query result:", result2)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
