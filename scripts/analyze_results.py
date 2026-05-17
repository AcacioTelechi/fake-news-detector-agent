"""CLI de análise dos resultados do pipeline.

Uso (a partir da raiz do repositório):
    python scripts/analyze_results.py                       # último JSONL de output/
    python scripts/analyze_results.py --input output/x.jsonl
    python scripts/analyze_results.py --top 25
    python scripts/analyze_results.py --with-engagement     # ranqueia por alcance
    python scripts/analyze_results.py --json                # resumo em JSON
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.analysis import (
    find_latest_results,
    load_results,
    summarize,
    top_fake_candidates,
)

DEFAULT_ENGAGEMENT_CSV = "input/2023_Completo_redem_0304.csv"


def _bar(n: int, total: int, width: int = 30) -> str:
    if not total:
        return ""
    fill = int(width * n / total)
    return "█" * fill + "·" * (width - fill)


def main() -> int:
    ap = argparse.ArgumentParser(description="Analisa resultados do detector de fake news")
    ap.add_argument("--input", default=None,
                    help="JSONL de resultados (default: mais recente em output/)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--with-engagement", nargs="?", const=DEFAULT_ENGAGEMENT_CSV,
                    default=None,
                    help="ranqueia candidatos por alcance usando o CSV de input")
    ap.add_argument("--json", action="store_true",
                    help="imprime o resumo em JSON e sai")
    args = ap.parse_args()

    path = args.input or find_latest_results(str(REPO_ROOT / "output"))
    if not path:
        print("Nenhum JSONL de resultados encontrado em output/.")
        return 1

    records = load_results(path)
    if not records:
        print(f"{path}: nenhum registro válido.")
        return 1

    summary = summarize(records)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    s = summary
    n = s["total"]
    print("=" * 64)
    print(f"ANÁLISE DE RESULTADOS — {path}")
    print("=" * 64)

    print(f"\n1. PROCESSAMENTO ({n:,} registros)")
    print(f"   success ........ {s['success']:,}")
    print(f"   com error ...... {s['errored']:,}")
    t = s["time_stats"]
    if t["mean_s"] is not None:
        print(f"   tempo .......... média {t['mean_s']}s | mediana "
              f"{t['median_s']}s | máx {t['max_s']}s | total {t['total_min']} min")

    print("\n2. RELEVÂNCIA (triagem do entry_node)")
    base = s["success"] or 1
    print(f"   relevant=true .. {s['relevant']:,} "
          f"({100*s['relevant']/base:.1f}%)  {_bar(s['relevant'], base)}")
    print(f"   relevant=false . {s['not_relevant']:,} "
          f"({100*s['not_relevant']/base:.1f}%)  {_bar(s['not_relevant'], base)}")

    inc = s["inconclusive"]
    print("\n3. INCONCLUSIVOS (não fabricaram score)")
    print(f"   total .......... {inc['total']:,}")
    print(f"   recusa/plano ... {inc['by_plan_refusal']:,}")
    print(f"   cota/pesquisa .. {inc['by_research_or_quota']:,}")
    print(f"   recusa explícita no texto (plan/justif): {s['refusals']:,}")

    print(f"\n4. SCORES REAIS ({s['scored']:,} análises concluídas)")
    st = s["score_stats"]
    if st["mean"] is not None:
        print(f"   média {st['mean']} | mediana {st['median']} | "
              f"min {st['min']} | max {st['max']}")
        for band, cnt in s["score_bands"].items():
            print(f"   {band:<26} {cnt:>6,}  {_bar(cnt, s['scored'])}")
    else:
        print("   (nenhum score numérico — provável estouro de cota)")

    if s["errors"]:
        print("\n5. ERROS (campo error + research_errors)")
        for cat, cnt in s["errors"].items():
            print(f"   {cat:<20} {cnt:,}")

    eng_map = None
    if args.with_engagement:
        from src.utils.dataset import load_posts
        print(f"\n[carregando engajamento de {args.with_engagement} …]")
        eng_map = {p.id: p.engagement for p in load_posts(args.with_engagement)}

    top = top_fake_candidates(records, n=args.top, engagement_by_id=eng_map)
    order = "score, depois alcance" if eng_map else "score"
    print(f"\n6. TOP {len(top)} CANDIDATOS A FAKE NEWS (por {order})")
    print("-" * 64)
    for c in top:
        eng = f" | alcance {c['engagement']:,}" if c["engagement"] is not None else ""
        print(f"[{c['id_mention']}] score={c['score']}{eng}")
        print(f"  texto: {c['text']!r}")
        print(f"  just.: {c['justification']}")
        print("-" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
