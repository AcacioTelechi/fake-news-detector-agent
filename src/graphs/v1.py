from langgraph.graph import StateGraph, END
from src.models import AgentState, RuntimeContext
from src.models.nodes import (
    entry_node,
    plan_node,
    research_node,
    analyst_node,
    inconclusive_node,
)


builder = StateGraph(state_schema=AgentState, context_schema=RuntimeContext)


builder.add_node("entry", entry_node)
builder.add_node("planner", plan_node)
builder.add_node("research", research_node)
builder.add_node("analyst", analyst_node)
builder.add_node("inconclusive", inconclusive_node)

builder.set_entry_point("entry")
builder.add_conditional_edges(
    "entry",
    lambda state: state.relevance_analysis is not None
    and state.relevance_analysis.relevant,
    {False: END, True: "planner"},
)

# Se o planner recusou ou devolveu plano vazio, não pesquisa nem inventa score
builder.add_conditional_edges(
    "planner",
    lambda state: state.plan_failed,
    {True: "inconclusive", False: "research"},
)

# Se a pesquisa não trouxe nenhuma evidência, encerra como inconclusivo
builder.add_conditional_edges(
    "research",
    lambda state: state.research_failed,
    {True: "inconclusive", False: "analyst"},
)

builder.add_edge("analyst", END)
builder.add_edge("inconclusive", END)

# graph = builder.compile(checkpointer=memory)
graph = builder.compile()
