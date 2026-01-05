import os
import json
from dotenv import load_dotenv

from src.graphs.v1 import graph
from langchain_openai import ChatOpenAI
from tavily import TavilyClient


load_dotenv()

if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "1"}}

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    runtime_context = {"model": model, "tavily": tavily}
    initial_state = {
        "post": "Cloroquina cura COVID",
        "max_revisions": 3,
    }
    resp = graph.invoke(initial_state, context=runtime_context)

    with open("resp.json", "w") as f:
        resp['response'] = resp['response'].model_dump()
        json.dump(resp, f)
