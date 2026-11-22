"""Fase 5 — Modelagem preditiva da Probabilidade de Default (PD).

* Divisão estratificada treino/teste (80/20) com semente fixa.
* Pré-processamento dentro de `sklearn.Pipeline` (imputação, escala, one-hot),
  ajustado apenas no treino → sem vazamento.
* Três candidatos: baseline ingênuo, Regressão Logística (padrão do setor,
  interpretável) e Gradient Boosting (HistGradientBoosting).
* Validação cruzada estratificada (5 folds) escolhe o campeão por ROC-AUC.
* Não usamos `class_weight` para manter a PD calibrada; o desbalanceamento é
  tratado na escolha do ponto de corte (fase de Resultados).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .. import context, db
from ..config import settings

log = logging.getLogger(__name__)

MODEL_LABELS = {
    "baseline": "Baseline (taxa média)",
    "logistic": "Regressão Logística",
    "gbm": "Gradient Boosting",
}


def feature_columns() -> list[str]:
    spec = context.spec()
    return [*spec.numeric_features, *spec.categorical_features]


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    spec = context.spec()
    return ColumnTransformer([
        ("num", Pipeline(num_steps), list(spec.numeric_features)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(spec.categorical_features)),
    ])


def candidates() -> dict[str, Pipeline]:
    rs = settings.random_state
    return {
        "baseline": Pipeline([("prep", build_preprocessor()), ("clf", DummyClassifier(strategy="prior"))]),
        "logistic": Pipeline([("prep", build_preprocessor()),
                              ("clf", LogisticRegression(max_iter=3000, C=1.0))]),
        "gbm": Pipeline([("prep", build_preprocessor(scale=False)),
                         ("clf", HistGradientBoostingClassifier(learning_rate=0.08, max_iter=350, max_leaf_nodes=31,
                                                                l2_regularization=1.0, early_stopping=True,
                                                                validation_fraction=0.1, random_state=rs))]),
    }


def ks_statistic(y_true: np.ndarray, proba: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(np.max(tpr - fpr))


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:  # recorte com uma só classe: métricas de ranking não existem
        nan = float("nan")
        return {"roc_auc": nan, "gini": nan, "ks": nan, "pr_auc": nan,
                "brier": float(brier_score_loss(y_true, proba)), "log_loss": nan,
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)), "f1": float(f1_score(y_true, pred, zero_division=0))}
    auc = roc_auc_score(y_true, proba)
    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, proba),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6))),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    spec = context.spec()
    X = df[feature_columns()].copy()
    # desfaz a imputação da limpeza: o pipeline imputa usando só o treino
    for flag, cols in spec.imputed.items():
        if flag in df:
            for col in cols:
                if col in X:
                    X.loc[df[flag] == 1, col] = np.nan
    return X, df[spec.target].astype(int)


@dataclass
class ModelingResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    importance: pd.DataFrame
    champion: str
    champion_id: str


def train(df: pd.DataFrame, run_id: str) -> ModelingResult:
    X, y = prepare_xy(df)
    X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
        X, y, df["record_id"], test_size=settings.test_size, stratify=y, random_state=settings.random_state)

    cv = StratifiedKFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_state)
    rows, fitted = [], {}
    preds = pd.DataFrame({"record_id": id_te.values, "y_true": y_te.values})
    for name, pipe in candidates().items():
        cvres = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=["roc_auc", "average_precision"], n_jobs=1)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        preds[f"p_{name}"] = proba
        m = evaluate(y_te.values, proba)
        m |= {"cv_auc_mean": float(cvres["test_roc_auc"].mean()), "cv_auc_std": float(cvres["test_roc_auc"].std()),
              "cv_pr_mean": float(cvres["test_average_precision"].mean()),
              "fit_seconds": float(cvres["fit_time"].mean())}
        rows.append({"model": name, "label": MODEL_LABELS[name], **m})
        fitted[name] = pipe
        log.info("Modelo %-9s AUC teste=%.4f  CV=%.4f±%.4f", name, m["roc_auc"], m["cv_auc_mean"], m["cv_auc_std"])

    metrics = pd.DataFrame(rows)
    champion = str(metrics.loc[metrics["model"] != "baseline"].sort_values("cv_auc_mean", ascending=False)["model"].iloc[0])
    metrics["is_champion"] = (metrics["model"] == champion).astype(int)

    # importância por permutação (agnóstica ao modelo) na amostra de teste
    sample = X_te.sample(n=min(4000, len(X_te)), random_state=settings.random_state)
    pi = permutation_importance(fitted[champion], sample, y_te.loc[sample.index], scoring="roc_auc",
                                n_repeats=5, random_state=settings.random_state, n_jobs=1)
    importance = pd.DataFrame({"feature": sample.columns, "importance": pi.importances_mean,
                               "std": pi.importances_std}).sort_values("importance", ascending=False)

    # registro de modelos (versionado)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    prev = db.read_sql("SELECT COALESCE(MAX(version), 0) AS v FROM model_registry")["v"].iloc[0]
    version = int(prev) + 1
    db.execute("UPDATE model_registry SET is_champion = 0")
    champion_id = ""
    for name, pipe in fitted.items():
        if name == "baseline":
            continue
        model_id = f"{name}-v{version}-{uuid.uuid4().hex[:6]}"
        path = settings.models_dir(context.current_id()) / f"{name}_v{version}.joblib"
        joblib.dump({"pipeline": pipe, "features": feature_columns(), "model_id": model_id, "dataset": context.current_id(),
                     "trained_at": now, "run_id": run_id}, path)
        m = metrics.loc[metrics["model"] == name].iloc[0].to_dict()
        params = {k: str(v) for k, v in pipe.named_steps["clf"].get_params().items()}
        db.execute(
            "INSERT INTO model_registry VALUES (:id,:run,:name,:ver,:path,:metrics,:params,:champ,:at)",
            id=model_id, run=run_id, name=name, ver=version, path=str(path), metrics=db.to_json(m),
            params=db.to_json(params), champ=int(name == champion), at=now)
        if name == champion:
            champion_id = model_id

    return ModelingResult(metrics, preds, importance, champion, champion_id)
