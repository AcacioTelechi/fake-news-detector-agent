import datetime
import os
import json
from dotenv import load_dotenv

from src.graphs.v1 import graph
from src.models.context import ModelsRegistry, ModelConfig
from tavily import TavilyClient
from src.utils.observability import (
    UsageMetadataCallbackHandler,
    set_callback_handler,
    print_metrics_summary,
)


load_dotenv()

if __name__ == "__main__":
    thread = {"configurable": {"thread_id": "1"}}

    # Get model name from environment variable or use default
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    # model_name = os.getenv("MODEL_NAME", "llama3.1:8b")
    # model_name = os.getenv("MODEL_NAME", "qwen2.5:1.5b")
    temperature = float(os.getenv("MODEL_TEMPERATURE", "0"))

    # Criar registry de modelos - pode usar modelos diferentes para cada node
    # Opção 1: Usar string (aplica default_temperature a todos)
    models_registry = ModelsRegistry(
        entry=model_name,
        planner=model_name,
        researcher=model_name,
        analyst=model_name,
        default_temperature=temperature,
    )
    
    # Opção 2: Usar ModelConfig para configurações específicas por node
    # models_registry = ModelsRegistry(
    #     entry=model_name,  # usa default_temperature
    #     planner=ModelConfig(model="gpt-4", temperature=0.7),
    #     researcher=ModelConfig(model="gpt-3.5-turbo", temperature=0.0),
    #     analyst=ModelConfig(
    #         model="ollama:llama2",
    #         temperature=0.3,
    #         model_provider="ollama",
    #         base_url="http://localhost:11434",
    #     ),
    #     default_temperature=temperature,
    # )

    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # Criar callback handler para rastrear tokens
    callback = UsageMetadataCallbackHandler()
    set_callback_handler(callback)

    # Posts
    posts = [
        "Bom dia!",
        "Cloroquina cura COVID",
        "Gastar dinheiro público com campanhas de incentivo ao aborto, ideologia de gênero ou troca de sexo de crianças? NÃO! ❌ Não vamos permitir!",
        "Dezesseis bilhões de reais para artistas em 2023. Em meio a aumentos de impostos e taxas abusivas para o povo, o desgoverno Lula mostrou quais são as suas prioridades.No governo Bolsonaro, mostramos que é possível fazer cultura sem precisar comprar ninguém. Demonstramos que o dinheiro pode, sim, chegar à ponta, aos artistas pequenos que, agora, voltaram a ser deixados de lado.",
    ]
    # Configurar callback no config do graph
    config = {
        "configurable": {"thread_id": "1"},
        "callbacks": [callback],
    }
    runtime_context = {"models_registry": models_registry, "tavily": tavily}

    resp_list = []
    for post in posts:
        initial_state = {
            "post": post,
            "max_revisions": 3,
        }
        resp = graph.invoke(initial_state, context=runtime_context, config=config)

        # Imprimir resumo de métricas
        print_metrics_summary(resp)

        resp["metrics"] = {k: v.model_dump() for k, v in resp["metrics"].items()}
        resp["relevance_analysis"] = resp["relevance_analysis"].model_dump()
        if resp["relevance_analysis"]["relevant"]:
            resp["response"] = resp["response"].model_dump()
        resp_list.append(resp)

    with open("outputs.json", "r", encoding="utf-8") as f:
        actual_ouputs = json.load(f)
    
    for resp in resp_list:
        resp['model'] = model_name
        resp['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        resp['temperature'] = temperature

        actual_ouputs.append(resp)

    with open("outputs.json", "w", encoding="utf-8") as f:
        json.dump(actual_ouputs, f, ensure_ascii=False)