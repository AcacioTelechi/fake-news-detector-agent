"""Teste determinístico do encanamento de queries do research_node.

Sem Ollama/Tavily: stubs verificam que o que o LLM devolve é sanitizado e
passado EXATAMENTE para tavily.search, e que generated/sent/received são
capturados para auditoria. Rode:  .venv/bin/python tests/test_research_wiring.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.nodes import research_node, _sanitize_queries
from src.models.schemas import Queries
from src.models.state import AgentState


class _Structured:
    def __init__(self, payload):
        self._payload = payload

    def invoke(self, messages):
        return self._payload


class _Model:
    def __init__(self, payload):
        self._payload = payload

    def with_structured_output(self, schema):
        return _Structured(self._payload)


class _Registry:
    def __init__(self, payload):
        self._m = _Model(payload)

    def get_model(self, name):
        return self._m

    def get_model_name(self, name):
        return "stub"


class _Tavily:
    def __init__(self):
        self.calls = []

    def search(self, query, max_results=2):
        self.calls.append((query, max_results))
        return {"query": query, "results": [{"content": f"RES::{query}"}]}


class _TavilyEmpty:
    """Responde 200 mas sem resultados (falha 'mole' do Tavily)."""

    def __init__(self):
        self.calls = []

    def search(self, query, max_results=2):
        self.calls.append((query, max_results))
        return {"query": query, "results": []}


class _TavilyRaising:
    """Lança como o Tavily faz em cota/rede."""

    def __init__(self):
        self.calls = []

    def search(self, query, max_results=2):
        self.calls.append((query, max_results))
        raise RuntimeError("Forbidden: exceeds your plan's set usage limit")


class _Ctx:
    pass


class _Runtime:
    def __init__(self, ctx):
        self.context = ctx


def _run(payload, plan="plano de verificação", tavily=None):
    tav = tavily if tavily is not None else _Tavily()
    ctx = _Ctx()
    ctx.models_registry = _Registry(payload)
    ctx.tavily = tav
    state = AgentState(post="post", plan=plan, content=[])
    out = research_node(state, _Runtime(ctx))
    return out, tav


def test_happy_path_and_sanitization():
    raw = [
        "  São Paulo eleição 2024  ",   # válida (será trimada)
        "",                              # vazia -> descartada
        "x",                             # 1 char -> descartada
        "Q" * 500,                       # > 400 -> truncada p/ 400
        '"Boulos laudo Marçal"',         # aspas removidas
    ]
    out, tav = _run(Queries(queries=raw))

    expected_sent = _sanitize_queries(raw)
    # sanidade independente do conteúdo esperado
    assert expected_sent == [
        "São Paulo eleição 2024",
        "Q" * 400,
        "Boulos laudo Marçal",
    ], expected_sent

    assert out["generated_queries"] == raw, "cru do LLM deve ser preservado"
    assert out["sent_queries"] == expected_sent, "sanitização incorreta"

    sent_to_tavily = [c[0] for c in tav.calls]
    assert sent_to_tavily == expected_sent, (
        f"o que chegou ao tavily.search != sent_queries: {sent_to_tavily}"
    )
    assert all(c[1] == 2 for c in tav.calls), "max_results deve ser 2"
    assert out["tavily_received_queries"] == expected_sent, "echo do Tavily divergente"
    assert out["content"] == [f"RES::{q}" for q in expected_sent]
    assert out["research_failed"] is False
    assert out["research_errors"] == []
    print("OK: geração -> sanitização -> tavily.search exatos; auditoria capturada")


def test_structured_output_failure_triggers_fallback():
    class _NoQueries:  # structured output quebrou: objeto sem .queries (caso "{ ")
        pass

    out, tav = _run(_NoQueries(), plan="x")
    assert out["generated_queries"] == [], out["generated_queries"]
    # fallback usa o post quando o LLM não dá query usável
    assert out["query_fallback_used"] is True
    assert out["sent_queries"] == ["post"], out["sent_queries"]
    assert [c[0] for c in tav.calls] == ["post"], tav.calls
    assert out["research_failed"] is False, "fallback obteve evidência"
    print("OK: structured output falho -> fallback pesquisa o post")


def test_all_queries_invalid_falls_back_to_post():
    out, tav = _run(Queries(queries=["", " ", "a"]), plan="x")
    assert out["generated_queries"] == ["", " ", "a"]
    assert out["query_fallback_used"] is True
    assert [c[0] for c in tav.calls] == ["post"]
    print("OK: todas inválidas -> fallback pesquisa o post")


def test_no_fallback_when_post_and_plan_unusable():
    tav = _Tavily()
    ctx = _Ctx()
    ctx.models_registry = _Registry(Queries(queries=[]))
    ctx.tavily = tav
    state = AgentState(post="", plan="", content=[])  # nada para o fallback
    out = research_node(state, _Runtime(ctx))
    assert out["sent_queries"] == []
    assert out["query_fallback_used"] is False
    assert tav.calls == [], "sem post nem plano: não chama o Tavily"
    assert out["research_failed"] is True, "honestamente inconclusivo"
    print("OK: sem post/plano -> sem fallback, research_failed=True (honesto)")


def test_tavily_trace_records_results_and_timing():
    out, _ = _run(Queries(queries=["São Paulo eleição 2024"]))
    tr = out["tavily_trace"]
    assert len(tr) == 1, tr
    assert tr[0]["query"] == "São Paulo eleição 2024"
    assert tr[0]["results"] == 1 and tr[0]["error"] is None
    assert isinstance(tr[0]["elapsed_s"], (int, float))
    print("OK: tavily_trace registra query, nº de resultados e tempo")


def test_tavily_empty_results_is_visible():
    out, _ = _run(Queries(queries=["query sem retorno"]), tavily=_TavilyEmpty())
    tr = out["tavily_trace"]
    assert tr[0]["results"] == 0 and tr[0]["error"] is None, tr
    assert out["research_failed"] is True
    assert out["research_errors"] == [], "0 resultados não é 'erro', mas fica visível na trace"
    print("OK: Tavily 200 com 0 resultados -> visível na trace (results=0)")


def test_tavily_exception_is_traced():
    out, _ = _run(Queries(queries=["q"]), tavily=_TavilyRaising())
    tr = out["tavily_trace"]
    assert tr[0]["results"] is None and "Forbidden" in tr[0]["error"], tr
    assert any("Forbidden" in e for e in out["research_errors"])
    assert out["research_failed"] is True
    print("OK: exceção do Tavily -> trace com erro + research_errors")


def test_debug_block_renders_tavily_line():
    from src.utils.batch import _debug_block
    out, _ = _run(Queries(queries=["São Paulo eleição 2024"]))
    rec = {
        "id_mention": "X", "success": True, "relevant": True,
        "full_text": "t", "generated_queries": out["generated_queries"],
        "sent_queries": out["sent_queries"],
        "tavily_received_queries": out["tavily_received_queries"],
        "tavily_trace": out["tavily_trace"], "metrics": {},
    }
    block = _debug_block(rec)
    assert "tavily: 'São Paulo eleição 2024'→1res(" in block, block
    assert "[total" in block
    print("OK: bloco de debug renderiza a linha 'tavily:' com res/tempo")


if __name__ == "__main__":
    test_happy_path_and_sanitization()
    test_structured_output_failure_triggers_fallback()
    test_all_queries_invalid_falls_back_to_post()
    test_no_fallback_when_post_and_plan_unusable()
    test_tavily_trace_records_results_and_timing()
    test_tavily_empty_results_is_visible()
    test_tavily_exception_is_traced()
    test_debug_block_renders_tavily_line()
    print("\nTODOS OS TESTES DE ENCANAMENTO PASSARAM")
