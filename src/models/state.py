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
    # Sinaliza que o planner se recusou a responder ou devolveu plano vazio
    plan_failed: bool = Field(default=False)
    # Sinaliza que a etapa de pesquisa (Tavily) não conseguiu nenhum resultado
    research_failed: bool = Field(default=False)
    # Resultado não pôde ser apurado (não confundir com score 0.0)
    inconclusive: bool = Field(default=False)
    # Mensagens de erro coletadas durante a pesquisa (não derrubam o pipeline)
    research_errors: List[str] = Field(default=[])
    # Auditoria das queries: cru do LLM, o que foi enviado, o que o Tavily ecoou
    generated_queries: List[str] = Field(default=[])
    sent_queries: List[str] = Field(default=[])
    tavily_received_queries: List[str] = Field(default=[])
