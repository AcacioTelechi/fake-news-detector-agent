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
    return {"plan": response.content}


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

    all_responses = []
    for q in queries.queries:
        response = runtime.context.tavily.search(query=q, max_results=2)
        for r in response["results"]:
            content.append(r["content"])
        all_responses.append(response)
    return {"content": content, "references": all_responses}


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
