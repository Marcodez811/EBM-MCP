import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json

load_dotenv()


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        self.session: Optional[ClientSession] = None
        # the resource manager that sets up the transport infrastructure underneath
        self.exit_stack = AsyncExitStack()
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = "gemini-2.5-flash"

    async def connect_to_server(self, server_path: str):
        """
        Connects the client to a MCP server (stdio protocol)
        Args:
            server_path : The path to the server script (either in .py or .js format)
        """
        is_python = server_path.endswith(".py")
        is_js = server_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_path], env=None
        )
        # This establishes the actual communication channel with the server using stdio.
        # The stdio_client function launches the server subprocess and returns transport objects for reading from and writing to it.
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        # get both the stdio and write streaming objects.
        self.stdio, self.write = stdio_transport
        # creates client-side MCP session based on the stdio streams provided
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        # initalization handshake (MCP)
        await self.session.initialize()
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def execute_tools(self, query: str) -> list[str]:
        """Process a query using Gemini and available tools"""
        # Get tools from MCP server
        list_tools_response = await self.session.list_tools()
        
        # Convert MCP tools to Gemini format
        function_declarations = [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema,
            )
            for tool in list_tools_response.tools
        ]
        
        tools = types.Tool(function_declarations=function_declarations)
        config = types.GenerateContentConfig(tools=[tools])

        response = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=query,
            config=config
        )
        
        # Check if there are function calls
        function_calls = [
            part.function_call 
            for part in response.candidates[0].content.parts 
            if part.function_call
        ]
        
        # If no function calls, return the text response directly
        if not function_calls:
            return response.text if response.text else "No response generated."
        
        # Handle function calls
        tool_responses = []
        for function_call in function_calls:
            tool_name = function_call.name
            tool_args = dict(function_call.args)
            result = await self.session.call_tool(tool_name, tool_args)
            tool_responses.append(result.content)
        
        return tool_responses

    async def pico_search(self, pico_data: dict[str, str], keys: list[str], num_results: int = 20, offset: int = 0):
        """Run PICO search using the pico data"""
        extracted_data = {key: pico_data[key] for key in keys}
        query = f"""
            Task: Perform a PubMed search based on the provided PICO data and return the results as a clean JSON array.

            PICO Data: {extracted_data}

            Instructions:
            1. Create a search query string by:
               - Combining keywords within each PICO category using OR operators
               - Connecting different categories using AND operators
               - Example: (diabetes OR "chronic right knee pain") AND ("knee replacement surgery" OR "open knee replacement" OR "minimally invasive knee replacement")
            
            2. Use the `search_pubmed_key_words` tool with:
               - The constructed query string
               - num_results: {num_results}
               - offset: {offset}
            
            3. Return ONLY a valid JSON array containing the article metadata. Each article should have:
               - pmid: string
               - title: string
               - doi: string
               - authors: string
               - journal: string
               - publication_date: string
               - abstract: string (full abstract)
            
            Return only the JSON array, no additional text or formatting.
        """
        results = await self.execute_tools(query)
        if results:
            search_results = list(map(lambda x: json.loads(x.text.replace("\n", "")), results[0]))
            return search_results
        return []
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")


    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        search_result = await client.pico_search(
            pico_data = {
                "P": ["diabetes", "chronic right knee pain"],
                "I": ["knee replacement surgery", "open knee replacement", "minimally invasive knee replacement"],
                "C": ["conservative management", "medication", "physiotherapy"],
                "O": ["pain relief", "functional recovery", "early return to work"]
            },
            keys = ["P", "I", "C"],
            num_results=5
        )
        print(json.dumps(search_result, indent=4))
        pmids = list(map(lambda item: item["PMID"], search_result))
        print(pmids)
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys
    asyncio.run(main())
