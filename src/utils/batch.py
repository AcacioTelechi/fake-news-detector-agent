"""Núcleo do batch runner: idempotência, detecção de cota e orquestração.

Independente de CLI. A camada de linha de comando vive em
`scripts/run_batch.py`.
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, List

from src.utils.dataset import Post

# Configuração default dos modelos (Ollama). Sobrescrevível pela CLI.
DEFAULT_OLLAMA_CONFIG = {
    "model_provider": "ollama",
    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
}
DEFAULT_MODELS_CONFIG = {
    "entry": "qwen2.5:1.5b",
    "planner": "llama3.1:8b",
    "researcher": "llama3.1:8b",
    "analyst": "llama3.1:8b",
}

# Marcadores de erro do Tavily que indicam cota esgotada (não falha transitória).
QUOTA_MARKERS = (
    "exceeds your plan's set usage limit",
    "Forbidden",
    "usage limit",
)


def is_quota_error(errors) -> bool:
    """True se algum erro de pesquisa indica cota do Tavily esgotada."""
    for e in errors or []:
        s = str(e)
        if any(m in s for m in QUOTA_MARKERS):
            return True
    return False


def compact_jsonl(jsonl_path: str) -> set:
    """Compacta o JSONL garantindo idempotência no resume.

    - descarta linhas corrompidas (escrita interrompida por kill);
    - mantém só o último registro por id_mention (dedupe);
    - reescreve atomicamente (tmp + os.replace).

    Retorna apenas os ids com resultado terminal (`success` verdadeiro —
    inclui inconclusivos, pois o grafo completou). Registros com
    `success=False` (falha de infra: modelo ausente, rede, etc.) NÃO entram
    no conjunto de processados: ficam no arquivo para auditoria mas serão
    reprocessados no próximo run, e o registro bom sobrescreve o ruim
    (último vence). Reexecutar sobre um arquivo todo-OK é no-op.
    """
    if not os.path.exists(jsonl_path):
        return set()

    by_id = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue  # linha truncada/corrompida — removida
            mid = rec.get("id_mention")
            if mid:
                by_id[mid] = rec  # último vence

    tmp = jsonl_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in by_id.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, jsonl_path)
    return {mid for mid, rec in by_id.items() if rec.get("success")}


def analyze_post(
    graph,
    tavily,
    post_id: str,
    post_text: str,
    *,
    models_config: dict = None,
    ollama_config: dict = None,
    temperature: float = 0.0,
) -> dict:
    """Roda o grafo para um post e devolve um registro plano de resultado."""
    from src.models.context import ModelsRegistry, ModelConfig
    from src.utils.observability import (
        UsageMetadataCallbackHandler,
        set_callback_handler,
    )

    models_config = models_config or DEFAULT_MODELS_CONFIG
    ollama_config = ollama_config or DEFAULT_OLLAMA_CONFIG

    callback = UsageMetadataCallbackHandler()
    set_callback_handler(callback)

    registry = ModelsRegistry(
        entry=ModelConfig(model=models_config["entry"], temperature=temperature, **ollama_config),
        planner=ModelConfig(model=models_config["planner"], temperature=temperature, **ollama_config),
        researcher=ModelConfig(model=models_config["researcher"], temperature=temperature, **ollama_config),
        analyst=ModelConfig(model=models_config["analyst"], temperature=temperature, **ollama_config),
        default_temperature=temperature,
    )
    config = {
        "configurable": {"thread_id": f"batch_{post_id}"},
        "callbacks": [callback],
    }
    runtime_context = {"models_registry": registry, "tavily": tavily}
    start = datetime.now()

    try:
        resp = graph.invoke(
            {"post": post_text, "max_revisions": 3},
            context=runtime_context,
            config=config,
        )
    except Exception as e:
        return {
            "id_mention": post_id, "full_text": post_text, "success": False,
            "error": f"{type(e).__name__}: {e}", "timestamp": start.isoformat(),
            "processing_time_s": (datetime.now() - start).total_seconds(),
            "models_config": models_config, "relevant": None,
            "relevance_reasoning": None, "plan": None, "score": None,
            "justification": None, "inconclusive": None,
            "research_failed": None, "research_errors": [], "metrics": {},
        }

    rel = resp.get("relevance_analysis")
    if hasattr(rel, "model_dump"):
        rel = rel.model_dump()
    rel = rel or {}

    response = resp.get("response")
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    response = response or {}

    metrics = {}
    for k, v in (resp.get("metrics") or {}).items():
        metrics[k] = v.model_dump() if hasattr(v, "model_dump") else v

    relevant = bool(rel.get("relevant"))
    return {
        "id_mention": post_id,
        "full_text": post_text,
        "success": True,
        "error": None,
        "timestamp": start.isoformat(),
        "processing_time_s": (datetime.now() - start).total_seconds(),
        "models_config": models_config,
        "relevant": relevant,
        "relevance_reasoning": rel.get("reasoning"),
        "plan": resp.get("plan") if relevant else None,
        "score": response.get("score") if relevant else None,
        "justification": response.get("justification") if relevant else None,
        "inconclusive": bool(resp.get("inconclusive")) if relevant else None,
        "research_failed": bool(resp.get("research_failed")) if relevant else None,
        "research_errors": resp.get("research_errors") or [],
        "metrics": metrics,
    }


def run_batches(
    posts: List[Post],
    out_jsonl: str,
    *,
    graph,
    tavily,
    batch_size: int = 1000,
    workers: int = 4,
    quota_error_limit: int = 5,
    analyze: Callable = analyze_post,
    models_config: dict = None,
    ollama_config: dict = None,
    temperature: float = 0.0,
    log: Callable[[str], None] = print,
) -> dict:
    """Processa `posts` (já ordenados) em lotes, com parada por cota.

    `posts` deve vir filtrado dos já processados (resume) pelo chamador.
    Retorna um dicionário-resumo.
    """
    n_batches = (len(posts) + batch_size - 1) // batch_size
    write_lock = threading.Lock()
    stop_event = threading.Event()
    quota_errors = done = relevant_n = inconclusive_n = errored_n = 0

    def _write(rec):
        with write_lock:
            with open(out_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    start_ts = datetime.now()
    for b in range(n_batches):
        if stop_event.is_set():
            break
        chunk = posts[b * batch_size:(b + 1) * batch_size]
        log(f"\n=== Lote {b + 1}/{n_batches} ({len(chunk)} posts) — "
            f"{datetime.now().strftime('%H:%M:%S')} ===")

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    analyze, graph, tavily, p.id, p.text,
                    models_config=models_config,
                    ollama_config=ollama_config,
                    temperature=temperature,
                ): p.id
                for p in chunk
            }
            for fut in as_completed(futures):
                pid = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {
                        "id_mention": pid, "success": False,
                        "error": f"Future error: {e}",
                        "timestamp": datetime.now().isoformat(),
                        "research_errors": [],
                    }
                _write(rec)
                done += 1
                if not rec.get("success"):
                    errored_n += 1
                if rec.get("relevant"):
                    relevant_n += 1
                if rec.get("inconclusive"):
                    inconclusive_n += 1
                if is_quota_error(rec.get("research_errors")):
                    quota_errors += 1
                    if quota_errors >= quota_error_limit and not stop_event.is_set():
                        stop_event.set()
                        log(f"\n[!] Cota do Tavily estourada "
                            f"({quota_errors} erros). Encerrando após drenar "
                            f"o lote atual…")
                if done % 50 == 0:
                    log(f"  {done:,} feitos | erros {errored_n} | "
                        f"relevantes {relevant_n} | "
                        f"inconclusivos {inconclusive_n} | "
                        f"quota-err {quota_errors}")
        if stop_event.is_set():
            break

    return {
        "processed": done,
        "errored": errored_n,
        "relevant": relevant_n,
        "inconclusive": inconclusive_n,
        "quota_errors": quota_errors,
        "stopped_by_quota": stop_event.is_set(),
        "duration_s": (datetime.now() - start_ts).total_seconds(),
        "out_jsonl": out_jsonl,
    }
