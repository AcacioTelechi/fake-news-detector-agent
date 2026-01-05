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

    # Posts
    posts = [
        "Cloroquina cura COVID",
        "Gastar dinheiro público com campanhas de incentivo ao aborto, ideologia de gênero ou troca de sexo de crianças? NÃO! ❌ Não vamos permitir!",
        "Dezesseis bilhões de reais para artistas em 2023. Em meio a aumentos de impostos e taxas abusivas para o povo, o desgoverno Lula mostrou quais são as suas prioridades.No governo Bolsonaro, mostramos que é possível fazer cultura sem precisar comprar ninguém. Demonstramos que o dinheiro pode, sim, chegar à ponta, aos artistas pequenos que, agora, voltaram a ser deixados de lado.",
    ]
    # Configurar callback no config do graph
    config = {
        "configurable": {"thread_id": "1"},
        "callbacks": [callback],
    }
    runtime_context = {"model": model, "tavily": tavily}

    resp_list = []
    for post in posts:
        initial_state = {
            "post": post,
            "max_revisions": 3,
        }
        resp = graph.invoke(initial_state, context=runtime_context, config=config)

        # Imprimir resumo de métricas
        print_metrics_summary(resp)

        resp["response"] = resp["response"].model_dump()
        resp["metrics"] = {k: v.model_dump() for k, v in resp["metrics"].items()}
        resp_list.append(resp)

    with open("resp.json", "w", encoding="utf-8") as f:
        json.dump(resp_list, f, ensure_ascii=False)
