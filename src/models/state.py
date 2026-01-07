from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

from typing import TypedDict, List, Dict, Optional
from src.models.schemas import RelevanceAnalysis, Response, Metrics


class AgentState(BaseModel):
    post: str = Field(default="")
    relevance_analysis: Optional[RelevanceAnalysis] = Field(default=None)
    plan: str = Field(default="")
    content: List[str] = Field(default=[])
    references: List[dict] = Field(default=[])
    response: Response = Field(default=Response(score=0.0, justification=""))
    revision_number: int = Field(default=0)
    max_revisions: int = Field(default=3)
    metrics: Optional[Dict[str, Metrics]] = Field(default={})
