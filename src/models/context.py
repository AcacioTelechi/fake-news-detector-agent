from pydantic import BaseModel, Field, ConfigDict
from langchain_core.language_models import BaseChatModel
from tavily import TavilyClient


class AgentContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model: BaseChatModel = Field(default=None)
    tavily: TavilyClient = Field(default=None)
