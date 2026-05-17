"""Utilitários.

Re-exports de `observability` são lazy (PEP 562) para que importar
módulos leves do pacote (ex.: `src.utils.dataset`) não puxe langchain.
"""

__all__ = [
    "UsageMetadataCallbackHandler",
    "track_node_metrics",
    "print_metrics_summary",
    "set_callback_handler",
    "get_callback_handler",
]


def __getattr__(name):
    if name in __all__:
        from src.utils import observability

        return getattr(observability, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
