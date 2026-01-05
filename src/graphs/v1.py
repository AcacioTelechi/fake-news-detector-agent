from langgraph.graph import StateGraph, END
from src.models import AgentState, AgentContext
from src.nodes.nodes import entry_node, plan_node, research_node, analyst_node


builder = StateGraph(state_schema=AgentState, context_schema=AgentContext)


builder.add_node("entry", entry_node)
builder.add_node("planner", plan_node)
builder.add_node("research", research_node)
builder.add_node("analyst", analyst_node)

builder.set_entry_point("entry")
builder.add_conditional_edges(
    "entry", lambda state: state["relevant"], {False: END, True: "planner"}
)

builder.add_edge("planner", "research")
builder.add_edge("research", "analyst")
builder.add_edge("analyst", END)

# graph = builder.compile(checkpointer=memory)
graph = builder.compile()
