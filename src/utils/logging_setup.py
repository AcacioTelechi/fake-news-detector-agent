"""Logger por execução: arquivo DEBUG detalhado + console INFO enxuto."""

import logging
import sys


def setup_logger(
    name: str,
    log_path: str,
    file_level: int = logging.DEBUG,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """Cria um logger que escreve detalhe (DEBUG) no arquivo e resumo no console.

    O arquivo abre em modo append (compatível com resume).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                      "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger
