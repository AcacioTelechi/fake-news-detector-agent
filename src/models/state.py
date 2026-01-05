from typing import TypedDict, List, Dict, Optional
from src.models.models import Response
from src.models.metrics import NodeMetrics


class AgentState(TypedDict):
    post: str
    relevant: bool
    plan: str
    content: List[str]
    response: Response
    revision_number: int = 0
    max_revisions: int = 3
    metrics: Optional[Dict[str, NodeMetrics]]
