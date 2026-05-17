"""Carga do dataset de posts para o pipeline de verificação.

Aceita tanto o schema bruto da base completa
(`Message-ID`, `Message`, `Number of Reactions, Comments & Shares`, ...)
quanto o schema já mapeado (`id_mention`, `full_text`). Devolve os posts
deduplicados por id (mantendo a ocorrência de maior engajamento) e
ordenados por engajamento decrescente.
"""

import csv
from typing import List, NamedTuple

csv.field_size_limit(10_000_000)

# nomes de coluna possíveis para cada campo
_ID_COLS = ("id_mention", "Message-ID")
_TEXT_COLS = ("full_text", "Message")
_ENGAGEMENT_COLS = (
    "Number of Reactions, Comments & Shares",
    "engagement",
)


class Post(NamedTuple):
    id: str
    text: str
    engagement: int


def parse_engagement(raw) -> int:
    """Converte o campo de engajamento (contagem) em int de forma robusta.

    A coluna é uma contagem inteira; eventuais separadores de milhar são
    descartados. Valores vazios/ inválidos viram 0.
    """
    if raw is None:
        return 0
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    return int(digits) if digits else 0


def _first(row: dict, names) -> str:
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return ""


def load_posts(path: str) -> List[Post]:
    """Lê o CSV (stream) e devolve posts únicos ordenados por engajamento desc.

    Dedupe por id mantendo a ocorrência de maior engajamento. Linhas sem id
    ou sem texto são ignoradas.
    """
    best = {}  # id -> Post (maior engajamento)
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if not (cols & set(_ID_COLS)) or not (cols & set(_TEXT_COLS)):
            raise ValueError(
                f"Schema inválido em {path}: precisa de uma coluna de id "
                f"({_ID_COLS}) e uma de texto ({_TEXT_COLS}). "
                f"Encontrado: {reader.fieldnames}"
            )
        for row in reader:
            pid = (_first(row, _ID_COLS) or "").strip()
            text = (_first(row, _TEXT_COLS) or "").strip()
            if not pid or not text:
                continue
            eng = parse_engagement(_first(row, _ENGAGEMENT_COLS))
            cur = best.get(pid)
            if cur is None or eng > cur.engagement:
                best[pid] = Post(pid, text, eng)

    return sorted(best.values(), key=lambda p: p.engagement, reverse=True)
