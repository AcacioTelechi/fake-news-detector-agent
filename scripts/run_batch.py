"""CLI do batch runner do detector de fake news.

Lê a base completa em `input/` (schema bruto ou mapeado), ordena por
engajamento, processa em lotes salvando incrementalmente em JSONL com
resume idempotente, e para ao confirmar estouro de cota do Tavily.

Uso (a partir da raiz do repositório):
    python scripts/run_batch.py                    # roda até estourar a cota
    python scripts/run_batch.py --limit 1000       # 1 lote para validar
    python scripts/run_batch.py --batch-size 500
    python scripts/run_batch.py --workers 4
    python scripts/run_batch.py --no-resume        # ignora progresso anterior
    python scripts/run_batch.py --dry-run          # só o plano de lotes
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# permite rodar de qualquer cwd: garante a raiz do repo no sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from src.utils.dataset import load_posts
from src.utils.batch import compact_jsonl, run_batches

DEFAULT_INPUT = "input/2023_Completo_redem_0304.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch runner do detector de fake news")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=int(os.getenv("MAX_WORKERS", "4")))
    ap.add_argument("--limit", type=int, default=None,
                    help="máximo de posts nesta execução")
    ap.add_argument("--quota-error-limit", type=int,
                    default=int(os.getenv("QUOTA_ERROR_LIMIT", "5")))
    ap.add_argument("--temperature", type=float,
                    default=float(os.getenv("MODEL_TEMPERATURE", "0")))
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="mostra o plano de lotes sem importar/rodar o grafo")
    args = ap.parse_args()

    load_dotenv()
    out_dir = REPO_ROOT / "output"
    out_dir.mkdir(exist_ok=True)

    posts = load_posts(args.input)
    print(f"Entrada: {args.input} ({len(posts):,} posts únicos, "
          f"ordenados por engajamento desc)")

    # Resume idempotente: reusa o JSONL mais recente e o compacta
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl = str(out_dir / f"batch_results_{run_id}.jsonl")
    processed = set()
    if not args.no_resume:
        prev = sorted(out_dir.glob("batch_results_*.jsonl"))
        if prev:
            out_jsonl = str(prev[-1])
            processed = compact_jsonl(out_jsonl)
            print(f"Resume: {len(processed):,} já processados em "
                  f"{out_jsonl} (JSONL compactado)")

    todo = [p for p in posts if p.id not in processed]
    if args.limit is not None:
        todo = todo[: args.limit]
    if not todo:
        print("Nada a processar.")
        return 0

    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    print(f"A processar: {len(todo):,} posts em {n_batches} lote(s) de "
          f"{args.batch_size} | {args.workers} workers")
    print(f"Saída: {out_jsonl}")
    print(f"Para por cota após {args.quota_error_limit} erros de Tavily.")

    if args.dry_run:
        for b in range(n_batches):
            chunk = todo[b * args.batch_size:(b + 1) * args.batch_size]
            print(f"  Lote {b + 1}/{n_batches}: {len(chunk)} posts "
                  f"(top eng. {chunk[0].engagement:,} … {chunk[-1].engagement:,})")
        return 0

    from src.graphs.v1 import graph
    from tavily import TavilyClient

    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    summary = run_batches(
        todo, out_jsonl,
        graph=graph, tavily=tavily,
        batch_size=args.batch_size,
        workers=args.workers,
        quota_error_limit=args.quota_error_limit,
        temperature=args.temperature,
    )

    print("\n" + "=" * 60)
    print(f"FIM. Processados nesta execução: {summary['processed']:,} "
          f"em {summary['duration_s']/60:.1f} min")
    print(f"  erros (success=False): {summary['errored']} | "
          f"relevantes: {summary['relevant']} | "
          f"inconclusivos: {summary['inconclusive']} | "
          f"erros de cota: {summary['quota_errors']}")
    print(f"  total acumulado (OK): {len(processed) + summary['processed'] - summary['errored']:,}")
    print(f"  JSONL: {summary['out_jsonl']}")
    if summary["errored"] == summary["processed"] and summary["processed"] > 0:
        print("  [!] TODOS falharam — verifique Ollama/modelos "
              "(ollama pull qwen2.5:1.5b llama3.1:8b) ou as chaves do .env. "
              "Os ids com erro serão reprocessados no próximo run (resume).")
    elif summary["errored"]:
        print(f"  [!] {summary['errored']} com erro de infra serão "
              "reprocessados no próximo run (resume).")
    if summary["stopped_by_quota"]:
        print("  Parada por estouro de cota do Tavily — "
              "rode de novo (resume) após renovar a cota.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
