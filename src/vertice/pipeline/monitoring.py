"""Fase 7 — Monitoramento: estabilidade da população (PSI).

PSI (Population Stability Index) compara a distribuição de uma variável entre
uma base de referência (treino) e uma base atual (teste, ou produção).
Regra de bolso usada no mercado de crédito:
    PSI < 0,10 estável · 0,10–0,25 atenção · > 0,25 mudança relevante
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference, current = reference.dropna(), current.dropna()
    if reference.dtype == object or str(reference.dtype) in ("str", "string") or reference.nunique() <= 12:
        ref = reference.astype(str).value_counts(normalize=True)
        cur = current.astype(str).value_counts(normalize=True)
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref = pd.cut(reference, edges).value_counts(normalize=True, sort=False)
        cur = pd.cut(current, edges).value_counts(normalize=True, sort=False)
    ref, cur = ref.align(cur, fill_value=0)
    ref, cur = ref.clip(lower=1e-4), cur.clip(lower=1e-4)
    return float(((cur - ref) * np.log(cur / ref)).sum())


def psi_status(value: float) -> str:
    if value < 0.10:
        return "Estável"
    if value < 0.25:
        return "Atenção"
    return "Mudança relevante"


def psi_table(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = [{"feature": f, "psi": round(psi(reference[f], current[f]), 5)} for f in features]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False)
    out["status"] = out["psi"].map(psi_status)
    return out
