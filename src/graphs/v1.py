from langgraph.graph import StateGraph

from src.models import AgentState, AgentContext
from src.nodes.nodes import plan_node, research_node, analyst_node


builder = StateGraph(state_schema=AgentState, context_schema=AgentContext)

builder.add_node("planner", plan_node)
builder.add_node("research", research_node)
builder.add_node("analyst", analyst_node)

builder.set_entry_point("planner")

# builder.add_conditional_edges(
#     "generate", should_continue, {END: END, "reflect": "reflect"}
# )

builder.add_edge("planner", "research")
builder.add_edge("research", "analyst")

# builder.add_edge("reflect", "research_critique")
# builder.add_edge("research_critique", "generate")

# graph = builder.compile(checkpointer=memory)
graph = builder.compile()
