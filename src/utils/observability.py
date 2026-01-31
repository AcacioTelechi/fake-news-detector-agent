import time
from functools import wraps
from typing import Callable, Any, Dict, List
from datetime import datetime
from threading import local
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from src.models.schemas import Metrics
from src.models.state import AgentState
from langgraph.runtime import Runtime
from src.models.context import RuntimeContext

# Thread-local storage para rastrear qual node está executando
_thread_local = local()


class UsageMetadataCallbackHandler(BaseCallbackHandler):
    """
    Callback handler para capturar usage_metadata de chamadas LLM.
    Armazena métricas de tokens por run_id.
    """

    def __init__(self):
        self.usage_metadata: List[Dict[str, Any]] = []
        self.current_node: str = None

    def on_llm_end(self, response: LLMResult, *, run_id, parent_run_id=None, **kwargs):
        """
        Chamado quando uma chamada LLM termina.
        Captura usage_metadata de cada geração.
        """
        for gen_list in response.generations:
            for gen in gen_list:
                msg = gen.message
                input_tokens = 0
                output_tokens = 0
                total_tokens = 0

                # Tentar extrair de usage_metadata (formato mais recente)
                if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                    usage = msg.usage_metadata
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)

                # Tentar extrair de response_metadata (formato antigo)
                elif (
                    hasattr(msg, "response_metadata")
                    and msg.response_metadata
                    and total_tokens == 0
                ):
                    metadata = msg.response_metadata
                    if "token_usage" in metadata:
                        usage = metadata["token_usage"]
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)

                # Adicionar apenas se houver tokens
                if total_tokens > 0:
                    self.usage_metadata.append(
                        {
                            "run_id": run_id,
                            "parent_run_id": parent_run_id,
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                            "node": self.current_node
                        }
                    )

    def get_tokens_for_node(self, node_name: str) -> Dict[str, int]:
        """
        Retorna tokens totais para um node específico.
        Soma todos os usage_metadata associados ao node.
        """
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        for usage in self.usage_metadata:
            if usage.get("node") == node_name:
                prompt_tokens += usage.get("input_tokens", 0)
                completion_tokens += usage.get("output_tokens", 0)
                total_tokens += usage.get("total_tokens", 0)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def set_current_node(self, node_name: str):
        """Define qual node está sendo executado atualmente."""
        self.current_node = node_name


def get_callback_handler() -> UsageMetadataCallbackHandler:
    """
    Obtém o callback handler do thread-local storage.
    O callback deve ser configurado antes da execução do grafo.
    """
    if not hasattr(_thread_local, "callback_handler"):
        return None
    return _thread_local.callback_handler


def set_callback_handler(handler: UsageMetadataCallbackHandler):
    """Define o callback handler no thread-local storage."""
    _thread_local.callback_handler = handler


def get_model_name(runtime: Runtime[RuntimeContext], node_name: str) -> str:
    """
    Extrai o nome do modelo base do runtime context para um node específico.
    
    Args:
        runtime: Runtime context contendo o ModelsRegistry
        node_name: Nome do node (entry, planner, researcher, analyst)
        
    Returns:
        Nome do modelo configurado para o node
    """
    if not runtime or not runtime.context or not runtime.context.models_registry:
        return "unknown"
    
    try:
        return runtime.context.models_registry.get_model_name(node_name)
    except (ValueError, AttributeError):
        return "unknown"


def track_node_metrics(node_name: str):
    """
    Decorator para rastrear métricas de execução de um nó.

    Mede tempo de execução, extrai uso de tokens via callback e armazena no state.
    Também loga métricas no console.
    """

    def decorator(func: Callable[[AgentState, Runtime[RuntimeContext]], Dict[str, Metrics]]):
        @wraps(func)
        def wrapper(
            state: AgentState, runtime: Runtime[RuntimeContext]
        ) -> Dict[str, Metrics]:
            # Obter callback handler se disponível
            callback_handler = get_callback_handler()

            # Definir node atual no callback handler
            if callback_handler:
                callback_handler.set_current_node(node_name)

            start_time = time.time()
            start_datetime = datetime.now().isoformat()

            # Executar o nó
            result = func(state, runtime)

            end_time = time.time()
            end_datetime = datetime.now().isoformat()
            execution_time = end_time - start_time

            # Inicializar métricas no state se não existir
            metrics = state.metrics

            # Extrair nome do modelo base
            base_model_name = get_model_name(runtime, node_name)

            # Criar métricas base (tempo e modelo)
            node_metrics = Metrics(
                base_model=base_model_name,
                execution_time=execution_time,
                start_time=start_datetime,
                end_time=end_datetime,
            )

            # Extrair tokens do callback handler
            if callback_handler:
                token_usage = callback_handler.get_tokens_for_node(node_name)
                node_metrics.prompt_tokens = token_usage.get("prompt_tokens", 0)
                node_metrics.completion_tokens = token_usage.get("completion_tokens", 0)
                node_metrics.total_tokens = token_usage.get("total_tokens", 0)

            # Armazenar métricas
            metrics[node_name] = node_metrics

            # # Log no console
            # print(f"[{node_name}] {node_metrics}")

            # Adicionar métricas ao resultado
            result["metrics"] = metrics

            return result

        return wrapper

    return decorator


def print_metrics_summary(state: dict[str, Any] | AgentState):
    """
    Imprime um resumo das métricas de todos os nodes.
    """
    metrics = state.get("metrics") if isinstance(state, dict) else state.metrics

    if not metrics:
        print("\n[Observability] Nenhuma métrica disponível.")
        return

    print("\n" + "=" * 60)
    print("RESUMO DE MÉTRICAS")
    print("=" * 60)

    total_time = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for node_name, node_metrics in metrics.items():
        print(f"\n[{node_name}]")
        print(f"  Tempo de execução: {node_metrics.formatted_time}")
        print(
            f"  Tokens: {node_metrics.total_tokens} "
            f"({node_metrics.prompt_tokens} prompt + {node_metrics.completion_tokens} completion)"
        )

        total_time += node_metrics.execution_time
        total_prompt_tokens += node_metrics.prompt_tokens
        total_completion_tokens += node_metrics.completion_tokens
        total_tokens += node_metrics.total_tokens

    print("\n" + "-" * 60)
    print("TOTAIS:")
    print(f"  Tempo total: {total_time:.2f}s")
    print(
        f"  Tokens totais: {total_tokens} ({total_prompt_tokens} prompt + {total_completion_tokens} completion)"
    )
    print("=" * 60 + "\n")
