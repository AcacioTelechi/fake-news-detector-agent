"""Análise dos resultados do pipeline (JSONL do batch runner).

Funções puras e testáveis; a CLI vive em `scripts/analyze_results.py`.
Tolera tanto o schema novo (com `inconclusive`/`research_errors`) quanto o
antigo (`output/analysis_results_*.jsonl`).
"""

import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import List, Optional

# Score sentinela usado pelo inconclusive_node (fora da faixa 0..1)
INCONCLUSIVE_SCORE = -1.0

# Linguagem de recusa do LLM (planner/analista se negando a responder)
REFUSAL_RE = re.compile(
    r"n[ãa]o\s+posso\s+(atender|cumprir|ajudar|fazer|realizar|continuar)"
    r"|pe[çc]o\s+desculpas,?\s+mas\s+n[ãa]o\s+posso"
    r"|i\s+(cannot|can't|am not able|am unable)"
    r"|as an ai\b",
    re.IGNORECASE,
)


def find_latest_results(output_dir: str = "output") -> Optional[str]:
    """Acha o JSONL mais recente (prioriza batch_results_, depois analysis_)."""
    d = Path(output_dir)
    for pattern in ("batch_results_*.jsonl", "analysis_results_*.jsonl"):
        files = sorted(d.glob(pattern))
        if files:
            return str(files[-1])
    return None


def load_results(path: str) -> List[dict]:
    """Lê o JSONL ignorando linhas corrompidas."""
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    return recs


def classify_error(msg: str) -> str:
    """Normaliza uma mensagem de erro do Tavily/pipeline em categoria."""
    s = str(msg)
    if "exceeds your plan" in s or "Forbidden" in s or "usage limit" in s:
        return "tavily_quota"
    if "too short" in s:
        return "query_too_short"
    if "too long" in s:
        return "query_too_long"
    if "Future error" in s:
        return "future_error"
    return "other"


def score_band(score: float) -> str:
    """Banda interpretável do score de fake news."""
    if score is None:
        return "sem_score"
    if score == INCONCLUSIVE_SCORE or score < 0:
        return "inconclusivo"
    if score >= 0.8:
        return "provável_fake (>=0.8)"
    if score >= 0.6:
        return "suspeito (0.6–0.8)"
    if score > 0.4:
        return "incerto (~0.5)"
    if score > 0.2:
        return "provável_verdade (0.2–0.4)"
    return "verdadeiro (<=0.2)"


def _has_refusal(rec: dict) -> bool:
    for k in ("plan", "justification", "relevance_reasoning"):
        v = rec.get(k)
        if v and REFUSAL_RE.search(str(v)):
            return True
    return False


def summarize(records: List[dict]) -> dict:
    """Calcula o resumo completo dos resultados."""
    n = len(records)
    success = [r for r in records if r.get("success")]
    errored = [r for r in records if r.get("error")]

    relevant = [r for r in success if r.get("relevant")]
    not_relevant = [r for r in success if r.get("relevant") is False]

    # Inconclusivos: flag explícita OU score sentinela
    inconclusive = [
        r for r in relevant
        if r.get("inconclusive") or r.get("score") == INCONCLUSIVE_SCORE
    ]
    # Por causa: plano recusado/vazio vs pesquisa/cota
    inc_by_plan = [r for r in inconclusive if not r.get("research_failed")]
    inc_by_research = [r for r in inconclusive if r.get("research_failed")]

    # Scores reais (relevante, não inconclusivo, score numérico válido)
    scored = [
        r for r in relevant
        if isinstance(r.get("score"), (int, float))
        and r.get("score") >= 0
        and not r.get("inconclusive")
    ]
    scores = [r["score"] for r in scored]

    # Erros: campo `error` + `research_errors` agregados
    err_counter = Counter()
    for r in records:
        if r.get("error"):
            err_counter[classify_error(r["error"])] += 1
        for e in r.get("research_errors") or []:
            err_counter[classify_error(e)] += 1

    band_counter = Counter(score_band(s) for s in scores)
    refusals = [r for r in records if _has_refusal(r)]
    times = [
        r["processing_time_s"] for r in records
        if isinstance(r.get("processing_time_s"), (int, float))
    ]

    summary = {
        "total": n,
        "success": len(success),
        "errored": len(errored),
        "relevant": len(relevant),
        "not_relevant": len(not_relevant),
        "inconclusive": {
            "total": len(inconclusive),
            "by_plan_refusal": len(inc_by_plan),
            "by_research_or_quota": len(inc_by_research),
        },
        "scored": len(scored),
        "score_stats": {
            "mean": round(statistics.mean(scores), 3) if scores else None,
            "median": round(statistics.median(scores), 3) if scores else None,
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
        },
        "score_bands": dict(band_counter.most_common()),
        "errors": dict(err_counter.most_common()),
        "refusals": len(refusals),
        "time_stats": {
            "mean_s": round(statistics.mean(times), 1) if times else None,
            "median_s": round(statistics.median(times), 1) if times else None,
            "max_s": round(max(times), 1) if times else None,
            "total_min": round(sum(times) / 60, 1) if times else None,
        },
    }
    return summary


def top_fake_candidates(
    records: List[dict],
    n: int = 15,
    engagement_by_id: Optional[dict] = None,
) -> List[dict]:
    """Maiores candidatos a fake news (score alto), opcionalmente por alcance.

    Se `engagement_by_id` for fornecido, ordena por (score, engajamento);
    caso contrário, só por score.
    """
    cands = [
        r for r in records
        if r.get("relevant")
        and isinstance(r.get("score"), (int, float))
        and r["score"] >= 0.5
        and not r.get("inconclusive")
    ]

    def key(r):
        eng = (engagement_by_id or {}).get(r.get("id_mention"), 0)
        return (r["score"], eng)

    cands.sort(key=key, reverse=True)
    out = []
    for r in cands[:n]:
        out.append({
            "id_mention": r.get("id_mention"),
            "score": r.get("score"),
            "engagement": (engagement_by_id or {}).get(r.get("id_mention")),
            "text": (r.get("full_text") or "")[:160],
            "justification": (r.get("justification") or "")[:240],
        })
    return out
