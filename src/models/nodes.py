import re

from src.models.prompts import (
    ENTRY_PROMPT,
    RESEARCHER_PROMPT,
    PLAN_PROMPT,
    ANALYST_PROMPT,
)
from src.models.state import AgentState
from src.models.context import RuntimeContext
from src.models.schemas import Queries, RelevanceAnalysis, Response
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from src.utils.observability import track_node_metrics


# Limites da API do Tavily
TAVILY_MIN_QUERY_LEN = 2
TAVILY_MAX_QUERY_LEN = 400

# Plano com menos que isso é tratado como recusa/vazio
MIN_PLAN_LEN = 40

# Frases típicas de recusa do LLM (planner se negando a analisar o conteúdo)
_REFUSAL_RE = re.compile(
    r"n[ãa]o\s+posso\s+(atender|cumprir|ajudar|fazer|realizar|continuar)"
    r"|pe[çc]o\s+desculpas,?\s+mas\s+n[ãa]o\s+posso"
    r"|i\s+(cannot|can't|am not able|am unable)"
    r"|as an ai\b",
    re.IGNORECASE,
)

# Plano que apenas se desculpa por não achar afirmações verificáveis
_NO_CLAIMS_RE = re.compile(
    r"^\s*(infelizmente|pe[çc]o desculpas|desculpe|sinto muito|lamento)\b",
    re.IGNORECASE,
)


def _plan_failed(plan: str) -> bool:
    """True se o planner recusou a tarefa ou não produziu um plano utilizável."""
    if not plan or not plan.strip():
        return True
    text = plan.strip()
    if len(text) < MIN_PLAN_LEN:
        return True
    if _REFUSAL_RE.search(text):
        return True
    if _NO_CLAIMS_RE.match(text):
        return True
    return False


def _sanitize_queries(queries: list[str]) -> list[str]:
    """Normaliza queries para os limites do Tavily, descartando as inválidas."""
    cleaned: list[str] = []
    for q in queries or []:
        if not isinstance(q, str):
            continue
        q = q.strip().strip('"').strip()
        if len(q) < TAVILY_MIN_QUERY_LEN:
            continue
        if len(q) > TAVILY_MAX_QUERY_LEN:
            q = q[:TAVILY_MAX_QUERY_LEN].rstrip()
        cleaned.append(q)
    return cleaned


@track_node_metrics("entry")
def entry_node(state: AgentState, runtime: Runtime[RuntimeContext]):
    messages = [SystemMessage(content=ENTRY_PROMPT), HumanMessage(content=state.post)]
    entry_model = runtime.context.models_registry.get_model("entry")
    response = entry_model.with_structured_output(RelevanceAnalysis).invoke(messages)
    return {"relevance_analysis": response}


@track_node_metrics("planner")
def plan_node(state: AgentState, runtime: Runtime[RuntimeContext]):
    messages = [SystemMessage(content=PLAN_PROMPT), HumanMessage(content=state.post)]
    planner_model = runtime.context.models_registry.get_model("planner")
    response = planner_model.invoke(messages)
    plan = response.content or ""
    return {"plan": plan, "plan_failed": _plan_failed(plan)}


@track_node_metrics("inconclusive")
def inconclusive_node(state: AgentState, runtime: Runtime[RuntimeContext]):
    """Encerra como inconclusivo em vez de fabricar um score.

    Disparado quando o planner se recusou/devolveu plano vazio ou quando a
    pesquisa não retornou nenhuma evidência. Usa score=-1.0 como sentinela
    (fora da faixa 0.0–1.0) para diferenciar de uma análise real.
    """
    if state.plan_failed:
        reason = (
            "Planner não produziu um plano verificável (recusa ou plano vazio); "
            "análise não realizada."
        )
    else:
        reason = (
            "Pesquisa não retornou evidências (limite do Tavily ou sem resultados); "
            "análise não realizada."
        )
    return {
        "response": Response(score=-1.0, justification=reason),
        "inconclusive": True,
    }


@track_node_metrics("researcher")
def research_node(state: AgentState, runtime: Runtime[RuntimeContext]):
    # Chamada LLM para gerar queries
    researcher_model = runtime.context.models_registry.get_model("researcher")
    queries_model = researcher_model.with_structured_output(Queries)
    queries: Queries = queries_model.invoke(
        [
            SystemMessage(content=RESEARCHER_PROMPT),
            HumanMessage(content=f"PLAN: {state.plan}"),
        ]
    )

    content = state.content
    if isinstance(content, str):
        content = [content]
    initial_len = len(content)

    safe_queries = _sanitize_queries(queries.queries)

    all_responses = []
    errors: list[str] = []
    for q in safe_queries:
        try:
            response = runtime.context.tavily.search(query=q, max_results=2)
        except Exception as e:  # quota, rede, query inválida — não derruba o pipeline
            errors.append(f"{type(e).__name__}: {e}")
            continue
        for r in response.get("results", []):
            content.append(r["content"])
        all_responses.append(response)

    research_failed = len(content) == initial_len  # nenhuma evidência nova obtida

    return {
        "content": content,
        "references": all_responses,
        "research_errors": errors,
        "research_failed": research_failed,
    }


@track_node_metrics("analyst")
def analyst_node(state: AgentState, runtime: Runtime[RuntimeContext]):
    content = "\n\n".join(state.content or [])

    user_message = HumanMessage(
        content=f"{state.post}\n\nHere is my plan:\n\n{state.plan}"
    )

    messages = [
        SystemMessage(content=ANALYST_PROMPT.format(content=content)),
        user_message,
    ]

    analyst_model = runtime.context.models_registry.get_model("analyst")
    structured_llm = analyst_model.with_structured_output(Response)
    response = structured_llm.invoke(messages)

    return {"response": response}
