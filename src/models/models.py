from pydantic import BaseModel
from typing import List

class Queries(BaseModel):
    queries: List[str]

class Response(BaseModel):
    score: float
    justification: str