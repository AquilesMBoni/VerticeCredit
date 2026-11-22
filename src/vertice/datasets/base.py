"""Contrato de uma base de dados: tudo o que a pipeline, a API e o dashboard
precisam saber sobre ela. Para adicionar uma base nova, crie um módulo que
devolva um `DatasetSpec` e registre em `datasets/__init__.py`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FormField:
    name: str
    label: str
    kind: str  # "int" | "float" | "select"
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: tuple = ()
    labels: dict | None = None


@dataclass(frozen=True)
class Rule:
    dimension: str
    description: str
    severity: str  # alta | média | baixa
    column: str = "—"


@dataclass(frozen=True)
class DatasetSpec:
    # identidade
    id: str
    name: str
    short: str
    currency: str
    raw_file: str
    is_synthetic: bool
    source_url: str
    source_label: str
    license: str
    source_html: str

    # leitura e dicionário
    read_raw: Callable[[Any], pd.DataFrame]
    expected_columns: tuple[str, ...]
    column_docs: dict[str, str]

    # papéis das colunas
    target: str
    grade: str
    grade_order: tuple[str, ...]
    amount: str
    income: str
    income_label: str
    rate: str
    rate_label: str
    purpose: str
    purpose_label: str
    age: str

    # validação
    build_schema: Callable[[], Any]
    rules: dict[str, Rule]
    validation_view: Callable[[pd.DataFrame], pd.DataFrame]

    # limpeza e atributos
    cleaning_steps: tuple[Callable, ...]
    add_features: Callable[[pd.DataFrame], pd.DataFrame]
    feature_docs: dict[str, str]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    imputed: dict[str, tuple[str, ...]]  # flag -> colunas que voltam a ser nulas na modelagem

    # modelagem e política
    policy_cols: tuple[str, ...]
    revenue_and_loss: Callable[[pd.DataFrame, float], tuple[np.ndarray, np.ndarray]]
    default_cutoff: float
    form_fields: tuple[FormField, ...]
    prepare_payload: Callable[[dict], pd.DataFrame]
    example_payload: dict
    predict_route: str

    # apresentação
    segment_dims: dict[str, str]
    dim_orders: dict[str, list[str]]
    dist_vars: dict[str, str]
    corr_cols: dict[str, str]
    impossible_vars: tuple[tuple[str, str], ...]
    compare_vars: dict[str, str]
    impute_view: tuple[str, str, str, str]  # (coluna, flag, título, rótulo do eixo)
    scatter: dict[str, Any]
    insights: Callable[[pd.DataFrame, pd.DataFrame], list[tuple[str, str]]]
    texts: dict[str, str] = field(default_factory=dict)

    def fmt_money(self, v: float, digits: int = 0) -> str:
        s = f"{v:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{self.currency} {s}"

    def grade_index(self, values: pd.Series) -> pd.Series:
        order = {g: i for i, g in enumerate(self.grade_order, start=1)}
        return values.astype(str).str.strip().str.upper().map(order)


@dataclass
class CleaningLog:
    """Registro auditável de cada regra de limpeza aplicada."""
    entries: list[dict] = field(default_factory=list)

    def add(self, step: str, rule: str, action: str, before: int, after: int, affected: int) -> None:
        self.entries.append(dict(order=len(self.entries) + 1, step=step, rule=rule, action=action,
                                 rows_before=before, rows_after=after, affected=int(affected)))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)
