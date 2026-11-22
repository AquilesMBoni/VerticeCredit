"""Fase 3 — Limpeza de dados.

As regras de cada base ficam em `vertice/datasets/<id>.py` (`spec.cleaning_steps`).
Cada regra é uma função pura registrada no `cleaning_log` com linhas antes,
depois e afetadas, para que qualquer pessoa audite o que mudou e por quê.

Valores imputados recebem uma coluna-flag. Na modelagem eles voltam a ser nulos
e são imputados dentro do pipeline do scikit-learn, ajustado só no treino, o
que evita vazamento de dados (data leakage).
"""
from __future__ import annotations

import logging

import pandas as pd

from .. import context
from ..datasets.base import CleaningLog

log = logging.getLogger(__name__)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.copy()
    lg = CleaningLog()
    for step in context.spec().cleaning_steps:
        df = step(df, lg)
    df = df.reset_index(drop=True)
    log.info("Limpeza [%s]: %s → %s linhas", context.current_id(), len(raw), len(df))
    return df, lg.to_frame()
