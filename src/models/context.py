from typing import TypedDict
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

class AgentContext(TypedDict):
    model: ChatOpenAI
    tavily: TavilyClient