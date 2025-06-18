from mcp_code_qa_server import RAGSystem

# Initialize and index a repository
rag_system = RAGSystem("grip-no-tests")
rag_system.index_repository()

# Ask questions
answer = rag_system.answer_question("What does the Server class do?")
print(answer)
