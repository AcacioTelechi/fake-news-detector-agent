from pydantic import BaseModel
from typing import List, Optional


class Queries(BaseModel):
    queries: List[str]

class RelevanceAnalysis(BaseModel):
    relevant: bool
    reasoning: str

class Response(BaseModel):
    score: float
    justification: str


class Metrics(BaseModel):
    """Métricas de execução de um nó do grafo"""

    execution_time: float  # em segundos
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None

    @property
    def formatted_time(self) -> str:
        """Retorna o tempo formatado"""
        return f"{self.execution_time:.2f}s"

    def __str__(self) -> str:
        """Representação string das métricas"""
        return (
            f"Execution time: {self.formatted_time} | "
            f"Tokens: {self.total_tokens} ({self.prompt_tokens} prompt + {self.completion_tokens} completion)"
        )
