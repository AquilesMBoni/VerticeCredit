"""Fase 4 — Engenharia de atributos e poder preditivo (WoE / IV).

Os atributos de cada base vêm de `spec.add_features`. WoE (Weight of Evidence)
e IV (Information Value) são a métrica clássica de modelos de crédito para
medir o quanto uma variável separa bons e maus pagadores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import context


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    return context.spec().add_features(df)


def _bins(series: pd.Series, q: int = 10) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(series) or series.nunique() <= 12:
        return series.astype(str)
    return pd.qcut(series, q=q, duplicates="drop").astype(str)


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "Irrelevante"
    if iv < 0.1:
        return "Fraco"
    if iv < 0.3:
        return "Médio"
    if iv < 0.5:
        return "Forte"
    return "Muito forte"


def woe_iv(df: pd.DataFrame, features: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (tabela WoE por faixa, tabela IV por variável)."""
    spec = context.spec()
    target = spec.target
    features = features or [*spec.numeric_features, *spec.categorical_features]
    total_bad = df[target].sum()
    total_good = len(df) - total_bad
    woe_rows, iv_rows = [], []
    for feat in features:
        tmp = pd.DataFrame({"bin": _bins(df[feat]), "y": df[target]})
        g = tmp.groupby("bin", observed=True)["y"].agg(n="size", bad="sum").reset_index()
        g["good"] = g["n"] - g["bad"]
        dist_good = (g["good"] + 0.5) / (total_good + 0.5 * len(g))  # suavização evita log(0)
        dist_bad = (g["bad"] + 0.5) / (total_bad + 0.5 * len(g))
        g["woe"] = np.log(dist_good / dist_bad)
        g["iv_part"] = (dist_good - dist_bad) * g["woe"]
        g["bad_rate"] = g["bad"] / g["n"]
        g.insert(0, "feature", feat)
        woe_rows.append(g)
        iv = float(g["iv_part"].sum())
        iv_rows.append({"feature": feat, "iv": round(iv, 4), "strength": iv_strength(iv), "n_bins": len(g)})
    woe = pd.concat(woe_rows, ignore_index=True)
    iv = pd.DataFrame(iv_rows).sort_values("iv", ascending=False).reset_index(drop=True)
    return woe, iv
