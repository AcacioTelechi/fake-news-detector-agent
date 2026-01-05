from typing import TypedDict, List
from src.models.models import Response


class AgentState(TypedDict):
    post: str
    plan: str
    content: List[str]
    response: Response
    revision_number: int = 0
    max_revisions: int = 3
