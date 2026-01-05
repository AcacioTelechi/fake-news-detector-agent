import os
import json
from dotenv import load_dotenv

from src.graphs.v1 import graph
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from src.utils.observability import (
    UsageMetadataCallbackHandler,
    set_callback_handler,
    print_metrics_summary,
)


load_dotenv()

if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "1"}}

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # Criar callback handler para rastrear tokens
    callback = UsageMetadataCallbackHandler()
    set_callback_handler(callback)


    # Configurar callback no config do graph
    config = {
        "configurable": {"thread_id": "1"},
        "callbacks": [callback],
    }
    runtime_context = {"model": model, "tavily": tavily}
    initial_state = {
        "post": "Cloroquina cura COVID",
        "max_revisions": 3,
    }

    resp = graph.invoke(initial_state, context=runtime_context, config=config)

    # Imprimir resumo de métricas
    print_metrics_summary(resp)

    with open("resp.json", "w", encoding="utf-8") as f:
        resp["response"] = resp["response"].model_dump()
        resp["metrics"] = {k: v.model_dump() for k, v in resp["metrics"].items()}
        json.dump(resp, f, ensure_ascii=False)
