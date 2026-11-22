"""Fase 1 — Ingestão.

Lê o arquivo bruto da base ativa, calcula a impressão digital (SHA-256) para
rastreabilidade e grava a cópia fiel em `raw_loans`. Nenhuma transformação de
conteúdo acontece aqui: a camada bruta é imutável.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .. import context, db
from ..config import settings

log = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    df: pd.DataFrame
    sha256: str
    size_bytes: int
    path: Path


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def raw_path() -> Path:
    return settings.raw_dir / context.spec().raw_file


def ingest(path: Path | None = None) -> IngestionResult:
    spec = context.spec()
    path = path or raw_path()
    if not path.exists() and spec.id == "br":
        from ..datasets import br_generator
        br_generator.main(path)  # a base brasileira é reprodutível: gera se não existir
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em {path}. Fonte: {spec.source_url}")
    df = spec.read_raw(path)
    missing = set(spec.expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no arquivo: {sorted(missing)}")

    df = df[list(spec.expected_columns)].copy()
    df.insert(0, "record_id", range(1, len(df) + 1))
    db.write_df(df, "raw_loans")
    sha = file_sha256(path)
    log.info("Ingestão [%s]: %s linhas, sha256=%s…", spec.id, len(df), sha[:12])
    return IngestionResult(df=df, sha256=sha, size_bytes=path.stat().st_size, path=path)
