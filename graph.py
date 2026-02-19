"""
LangGraph Agent for MCP Tools.
MCP Agent compatible with agent-chat-ui.
"""

from tool.getroadnetwork import *

import asyncio
import json
import logging
import os
from typing import Any, Dict, List
from langgraph.prebuilt import create_react_agent
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables
load_dotenv('E:\BaiduDisk\LangChain公开课\SumoManus\lang_ui\environment.env')

class Configuration:
    """Load and hold application configuration."""
    def __init__(self) -> None:
        self.api_key: str = os.getenv("LLM_API_KEY") or ""
        self.langsmith_api_key: str = os.getenv("LANGCHAIN_API_KEY")
        self.base_url: str | None = os.getenv("BASE_URL")
        self.model: str = os.getenv("MODEL") or "deepseek-chat"
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found. Please set it in your .env file.")

    @staticmethod
    def load_servers(file_path: str = "servers_config.json") -> Dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f).get("mcpServers", {})
        except FileNotFoundError:
            logging.warning(f"Config file {file_path} not found; using empty config.")
            return {}
        except json.JSONDecodeError as e:
            logging.error(f"JSON parse error in config file {file_path}: {e}")
            return {}


def create_agent():

    from langchain_tavily import TavilySearch
    search_tool = TavilySearch(max_results=5, topic="general", api_key=os.getenv("TAVILY_API_KEY"))
    tools = [create_sumo_net_from_bbox, search_tool]
    """Create and return a LangGraph ReAct agent."""
    cfg = Configuration()

    # Set environment variables for the model client
    os.environ["DEEPSEEK_API_KEY"] = cfg.api_key
    if cfg.base_url:
        os.environ["DEEPSEEK_API_BASE"] = cfg.base_url

    # Initialise the LLM
    model = ChatDeepSeek(model="deepseek-chat", api_key=cfg.api_key)
    from langgraph.prebuilt import create_react_agent

    # Load system prompt
    try:
        with open("agent_prompts.txt", "r", encoding="utf-8") as f:
            prompt = f.read()
    except FileNotFoundError:
        prompt = "You are a helpful assistant that can assist users with various tasks."

    agent = create_react_agent(model=model, tools=tools, prompt=prompt)

    return agent

# # Create and export the graph object required by agent-chat-ui
# graph = asyncio.run(create_agent())

# # Function for running an interactive chat loop (for local testing)
# async def run_chat_loop():
#     """Command-line chat loop (for testing)."""
#     agent = await create_agent()
#
#     print("\nMCP Agent started. Type 'quit' to exit.")
#
#     config = {"configurable": {"thread_id": "1"}}
#
#     while True:
#         user_input = input("\nYou: ").strip()
#         if user_input.lower() == "quit":
#             break
#
#         try:
#             result = await agent.ainvoke(
#                 {"messages": [{"role": "user", "content": user_input}]},
#                 config
#             )
#             print(f"\nAI: {result['messages'][-1].content}")
#         except Exception as exc:
#             print(f"\nError: {exc}")

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
#     asyncio.run(run_chat_loop())
