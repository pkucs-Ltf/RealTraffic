"""
LangGraph Agent for MCP Tools
适配 agent-chat-ui 的 MCP Agent
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

# 加载环境变量
load_dotenv('E:\BaiduDisk\LangChain公开课\SumoManus\lang_ui\environment.env')

class Configuration:
    """读取配置"""
    def __init__(self) -> None:
        self.api_key: str = os.getenv("LLM_API_KEY") or ""
        self.langsmith_api_key: str = os.getenv("LANGCHAIN_API_KEY") 
        self.base_url: str | None = os.getenv("BASE_URL")
        self.model: str = os.getenv("MODEL") or "deepseek-chat"
        if not self.api_key:
            raise ValueError("❌ 未找到 LLM_API_KEY，请在 .env 中配置")

    @staticmethod
    def load_servers(file_path: str = "servers_config.json") -> Dict[str, Any]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f).get("mcpServers", {})
        except FileNotFoundError:
            logging.warning(f"⚠️ 配置文件 {file_path} 未找到，使用空配置")
            return {}
        except json.JSONDecodeError as e:
            logging.error(f"❌ 配置文件 {file_path} JSON 格式错误: {e}")
            return {}


def create_agent():
    
    
    from langchain_tavily import TavilySearch
    search_tool = TavilySearch(max_results=5, topic="general", api_key=os.getenv("TAVILY_API_KEY"))
    tools = [create_sumo_net_from_bbox,search_tool]
    """创建 LangGraph agent"""
    cfg = Configuration()
    
    # 设置环境变量
    os.environ["DEEPSEEK_API_KEY"] = cfg.api_key
    if cfg.base_url:
        os.environ["DEEPSEEK_API_BASE"] = cfg.base_url
    
    
    
    # 初始化模型
    model = ChatDeepSeek(model="deepseek-chat", api_key=cfg.api_key)
    from langgraph.prebuilt import create_react_agent
    
    
    # 读取提示词
    try:
        with open("agent_prompts.txt", "r", encoding="utf-8") as f:
            prompt = f.read()
    except FileNotFoundError:
        prompt = "你是一个智能助手，可以帮助用户处理各种任务。"
    
    
    agent = create_react_agent(model=model, tools=tools, prompt=prompt)

    
    return agent

# # 创建并导出 graph 对象 - 这是 agent-chat-ui 需要的
# graph = asyncio.run(create_agent())

# # 用于直接运行的函数
# async def run_chat_loop():
#     """命令行聊天循环（用于测试）"""
#     agent = await create_agent()
    
#     print("\n🤖 MCP Agent 已启动，输入 'quit' 退出")
    
#     config = {"configurable": {"thread_id": "1"}}
    
#     while True:
#         user_input = input("\n你: ").strip()
#         if user_input.lower() == "quit":
#             break
        
#         try:
#             result = await agent.ainvoke(
#                 {"messages": [{"role": "user", "content": user_input}]},
#                 config
#             )
#             print(f"\nAI: {result['messages'][-1].content}")
#         except Exception as exc:
#             print(f"\n⚠️  出错: {exc}")

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
#     asyncio.run(run_chat_loop())