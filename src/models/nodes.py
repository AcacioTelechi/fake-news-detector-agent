from src.models.prompts import (
    ENTRY_PROMPT,
    RESEARCHER_PROMPT,
    PLAN_PROMPT,
    ANALYST_PROMPT,
)
from src.models.state import AgentState
from src.models.context import AgentContext
from src.models.schemas import Queries, Response
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from src.utils.observability import track_node_metrics


@track_node_metrics("entry")
def entry_node(state: AgentState, runtime: Runtime[AgentContext]):
    messages = [SystemMessage(content=ENTRY_PROMPT), HumanMessage(content=state.post)]
    response = runtime.context.model.invoke(messages)
    return {
        "relevant": response.content.lower()
        in ["sim", "s", "yes", "y", "true", "t", "1"]
    }


@track_node_metrics("planner")
def plan_node(state: AgentState, runtime: Runtime[AgentContext]):
    messages = [SystemMessage(content=PLAN_PROMPT), HumanMessage(content=state.post)]
    response = runtime.context.model.invoke(messages)
    return {"plan": response.content}


@track_node_metrics("research")
def research_node(state: AgentState, runtime: Runtime[AgentContext]):
    # Chamada LLM para gerar queries
    structured_llm = runtime.context.model.with_structured_output(Queries)
    queries: Queries = structured_llm.invoke(
        [
            SystemMessage(content=RESEARCHER_PROMPT),
            HumanMessage(content=f"POST: {state.post} \n\n PLAN: {state.plan}"),
        ]
    )

    content = state.content
    if isinstance(content, str):
        content = [content]
    for q in queries.queries:
        response = runtime.context.tavily.search(query=q, max_results=2)
        for r in response["results"]:
            content.append(r["content"])

    return {"content": content}


@track_node_metrics("analyst")
def analyst_node(state: AgentState, runtime: Runtime[AgentContext]):
    content = "\n\n".join(state.content or [])

    user_message = HumanMessage(
        content=f"{state.post}\n\nHere is my plan:\n\n{state.plan}"
    )

    messages = [
        SystemMessage(content=ANALYST_PROMPT.format(content=content)),
        user_message,
    ]

    structured_llm = runtime.context.model.with_structured_output(Response)
    response = structured_llm.invoke(messages)

    return {"response": response}
