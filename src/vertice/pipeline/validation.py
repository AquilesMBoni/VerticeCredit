"""Fase 2 — Validação e qualidade de dados (contrato com Pandera).

Cada base declara o próprio contrato (`spec.build_schema`) e o catálogo de
regras (`spec.rules`), cada uma ligada a uma dimensão de qualidade. O mesmo
contrato roda no bruto (diagnóstico) e no limpo (garantia).
"""
from __future__ import annotations

import logging

import pandas as pd
import pandera.pandas as pa

from .. import context

log = logging.getLogger(__name__)


def run_quality_checks(df: pd.DataFrame, stage: str) -> tuple[pd.DataFrame, float]:
    """Executa o contrato e devolve (relatório por regra, score 0–100 = % de linhas sem violação)."""
    spec = context.spec()
    schema = spec.build_schema()
    view = spec.validation_view(df.copy())
    cols = [c for c in view.columns if c in schema.columns]
    try:
        schema.validate(view, lazy=True)
        failures = pd.DataFrame(columns=["column", "check", "index"])
    except pa.errors.SchemaErrors as exc:
        failures = exc.failure_cases.copy()
    failures["check"] = failures["check"].astype(str)

    rows = []
    for name, rule in spec.rules.items():
        if name == "not_nullable":
            nulls = [(c, int(view[c].isna().sum())) for c in cols
                     if not schema.columns[c].nullable and view[c].isna().any()]
            for col, n in nulls:
                rows.append((stage, name, col, rule.dimension, f"{rule.description} em {col}", rule.severity, n))
            if not nulls:
                rows.append((stage, name, "(todas)", rule.dimension, rule.description, rule.severity, 0))
            continue
        subset = failures[failures["check"] == name]
        n_failed = int(subset["index"].nunique()) if not subset.empty else 0
        rows.append((stage, name, rule.column, rule.dimension, rule.description, rule.severity, n_failed))

    report = pd.DataFrame(rows, columns=["stage", "rule", "column", "dimension", "description", "severity", "failed_rows"])
    report["failed_pct"] = report["failed_rows"] / max(len(df), 1) * 100
    report["status"] = report["failed_rows"].map(lambda n: "aprovado" if n == 0 else "reprovado")

    required = [c for c in cols if not schema.columns[c].nullable]
    bad_rows = set(failures["index"].dropna().unique()) | set(view.index[view[required].isna().any(axis=1)])
    score = round(100 * (1 - len(bad_rows) / max(len(df), 1)), 2)
    log.info("Qualidade [%s/%s]: score=%.2f, reprovadas=%s", spec.id, stage, score, int((report.status == "reprovado").sum()))
    return report, score


def dimension_scores(report: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Score por dimensão = 100 − % de linhas afetadas pela pior regra da dimensão."""
    g = report.groupby(["stage", "dimension"], as_index=False)["failed_rows"].max()
    g["score"] = (100 * (1 - g["failed_rows"] / max(n_rows, 1))).clip(lower=0).round(2)
    return g
