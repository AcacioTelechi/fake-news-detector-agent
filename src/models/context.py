from pydantic import BaseModel, Field, ConfigDict
from langchain_openai import ChatOpenAI
from tavily import TavilyClient


class AgentContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model: ChatOpenAI = Field(default=None)
    tavily: TavilyClient = Field(default=None)
