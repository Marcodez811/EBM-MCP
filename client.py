import asyncio
from contextlib import AsyncExitStack
from collections import Counter
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
        self.sessions: dict[str, ClientSession] = {}
        self.transports: dict[str, tuple] = {}
        # the resource manager that sets up the transport infrastructure underneath
        self.exit_stack = AsyncExitStack()
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_name = "gemini-2.5-flash"

    async def connect_to_server(self, server_name: str, server_path: str):
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
        stdio, write = stdio_transport
        self.transports[server_name] = (stdio, write)
        # creates client-side MCP session based on the stdio streams provided
        self.sessions[server_name] = await self.exit_stack.enter_async_context(
            ClientSession(self.transports[server_name][0], self.transports[server_name][1])
        )
        # initalization handshake (MCP)
        await self.sessions[server_name].initialize()
        response = await self.sessions[server_name].list_tools()
        tools = response.tools
        print(f"\nConnected to server '{server_name}' with tools:", [tool.name for tool in tools])

    async def execute_tools(self, query: str) -> list[str]:
        """Process a query using Gemini and available tools"""
        # Aggregate tools from all servers and create a mapping
        all_tools = []
        tool_to_session = {}  # Maps tool name to the session that provides it
        
        for session in self.sessions.values():
            response = await session.list_tools()
            for tool in response.tools:
                # Check for tool name conflicts
                if tool.name in tool_to_session:
                    print(f"Warning: Tool '{tool.name}' exists in multiple servers. Using first occurrence.")
                else:
                    all_tools.append(tool)
                    tool_to_session[tool.name] = session
        
        # Convert MCP tools to Gemini format
        function_declarations = [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema,
            )
            for tool in all_tools
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
        
        # Handle function calls - route to the correct server
        tool_responses = []
        for function_call in function_calls:
            tool_name = function_call.name
            tool_args = dict(function_call.args)
            
            # Look up which session has this tool
            if tool_name in tool_to_session:
                session = tool_to_session[tool_name]
                result = await session.call_tool(tool_name, tool_args)
                tool_responses.append(result.content)
            else:
                print(f"Error: Tool '{tool_name}' not found in any connected server")
        
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

    async def export_publications(self, ids: list[str]):
        """
        Returns the annotation metadata of the search result to the user.
        """
        query = f"""
            Task: Use the export_publications tool to export the pubtator data for the provided list of PubMed IDs.
            
            Parameters:
            - pmids: {ids}
            - format: "biocjson"
            
            Return the raw JSON data from the tool without any additional formatting or explanation.
        """
        classified_result = await self.execute_tools(query)
        if classified_result:
            return list(map(lambda x: json.loads(x.text.replace("\n", "")), classified_result[0]))
        return []

    async def analyze_documents(self, pmids: list[str], data: dict[str, dict]):
        """
        Traverses the nested JSON structure to count all annotation types.
        """
        document_counter = {}
        for pmid in pmids:
            document_counter[pmid] = Counter()

        documents = data['PubTator3']
        # Loop through each document in the file
        for doc in documents:
            pubmed_id = doc.get('id', "")
            assert pubmed_id, "Not a valid document!"
            passages = doc.get('passages', [])
            # For each passage of a document (title, abstract)
            for passage in passages:
                # Access the 'annotations' list
                annotations = passage.get('annotations', [])
                # Loop through each annotation
                for annotation in annotations:
                    # Access the 'type' from the 'infons' object
                    if 'infons' in annotation and 'type' in annotation['infons']:
                        annotation_type = annotation['infons']['type']
                        document_counter[pubmed_id][annotation_type] += 1
        
        return document_counter

    
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
        print("Usage: python client.py <server_name:path_to_server_script> ...")
        print("Example: python client.py pubmed:./PubMed-MCP-Server/pubmed_server.py pubtator:./PubTator-MCP-Server/PubTator_server.py")
        sys.exit(1)

    client = MCPClient()
    try:
        # Parse server arguments in format "name:path"
        for arg in sys.argv[1:]:
            if ':' in arg:
                server_name, server_path = arg.split(':', 1)
            else:
                # If no name provided, use filename as name
                server_path = arg
                server_name = os.path.basename(server_path).replace('.py', '').replace('.js', '')
            
            await client.connect_to_server(server_name, server_path)
        search_result = await client.pico_search(
            pico_data = {
                "P": ["adults with osteoarthritis", "patients with chronic low back pain", "elderly patients at risk for falls"],
                "I": ["acupuncture", "exercise therapy", "use of assistive devices"],
                "C": ["standard pain medication", "placebo", "usual care"],
                "O": ["improved pain scores", "increased mobility", "reduced incidence of falls"]
            },
            keys = ["P", "I", "C"],
            num_results=20
        )
        pmids = list(map(lambda item: item["PMID"] if "PMID" in item.keys() else item["pmid"], search_result))
        # id_to_doc = list(map(lambda item: {item["PMID"] if "PMID" in item.keys() else item["pmid"]: item}, search_result))
        if not pmids:
            return
        pubtator_output = await client.export_publications(pmids)
        if not pubtator_output:
            return
        all_counts = await client.analyze_documents(pmids, pubtator_output[0])

        # Create a classified dictionary: {"Disease": [list of pmids], "Species": [list of pmids], ...}
        # Only include documents where the annotation type is at least 33% of all annotations
        classified_dict = {}
        threshold = 0.3
        
        for doc_id, counter in all_counts.items():
            total_annotations = sum(counter.values())
            if total_annotations == 0:
                continue
            
            for annotation_type, count in counter.items():
                percentage = count / total_annotations
                if percentage >= threshold:
                    if annotation_type not in classified_dict:
                        classified_dict[annotation_type] = []
                    classified_dict[annotation_type].append(doc_id)
        
        print("\nClassified Results:")
        print(json.dumps(classified_dict, indent=4))

    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys
    asyncio.run(main())
