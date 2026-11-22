"""Orquestrador da pipeline.

Executa as fases em ordem, mede o tempo de cada uma, grava artefatos no SQLite
e registra a execução em `pipeline_runs`. Pode ser chamado pela CLI, pela API
(POST /pipeline/run) ou pelo dashboard.

    python -m vertice.pipeline.runner              # as duas bases
    python -m vertice.pipeline.runner --dataset br # só a brasileira
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd

from .. import context, db
from ..config import settings
from . import cleaning, features, ingestion, modeling, monitoring, validation

log = logging.getLogger("vertice.pipeline")

ProgressFn = Callable[[str, int, int], None]

STEPS = ["Ingestão", "Validação (bruto)", "Limpeza", "Validação (limpo)", "Atributos e IV",
         "Modelagem", "Monitoramento (PSI)", "Exportação"]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run(progress: ProgressFn | None = None, dataset: str | None = None) -> dict:
    if dataset is not None:
        with context.use(dataset):
            return _run(progress)
    return _run(progress)


def _run(progress: ProgressFn | None = None) -> dict:
    spec = context.spec()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    started = time.perf_counter()
    db.execute("INSERT INTO pipeline_runs (run_id, started_at, status) VALUES (:r, :s, 'executando')",
               r=run_id, s=_now())
    timings: list[tuple[str, float, int]] = []
    state: dict = {}

    def step(i: int, fn: Callable[[], int]) -> None:
        name = STEPS[i]
        if progress:
            progress(name, i, len(STEPS))
        t0 = time.perf_counter()
        rows_out = fn()
        timings.append((name, time.perf_counter() - t0, rows_out))
        log.info("✔ %-22s %.2fs", name, timings[-1][1])

    try:
        def _ingest():
            res = ingestion.ingest()
            state["raw"], state["sha"] = res.df, res.sha256
            return len(res.df)

        def _validate_raw():
            rep, score = validation.run_quality_checks(state["raw"], "bruto")
            state["q_raw"], state["score_raw"] = rep, score
            return len(state["raw"])

        def _clean():
            clean, logdf = cleaning.clean(state["raw"])
            logdf.insert(0, "run_id", run_id)
            db.write_df(logdf, "cleaning_log")
            state["clean"] = clean
            return len(clean)

        def _validate_clean():
            rep, score = validation.run_quality_checks(state["clean"], "limpo")
            state["score_clean"] = score
            report = pd.concat([state["q_raw"], rep], ignore_index=True)
            db.write_df(report, "quality_checks")
            dims = pd.concat([validation.dimension_scores(state["q_raw"], len(state["raw"])),
                              validation.dimension_scores(rep, len(state["clean"]))])
            db.write_df(dims, "quality_dimensions")
            return len(state["clean"])

        def _features():
            df = features.add_features(state["clean"])
            woe, iv = features.woe_iv(df)
            db.write_df(df, "clean_loans")
            db.write_df(woe, "feature_woe")
            db.write_df(iv, "feature_iv")
            state["clean"] = df
            return len(df)

        def _model():
            res = modeling.train(state["clean"], run_id)
            df = state["clean"]
            preds = res.predictions.merge(df[["record_id", *spec.policy_cols]], on="record_id")
            db.write_df(preds, "test_predictions")
            db.write_df(res.metrics, "model_metrics")
            db.write_df(res.importance, "feature_importance")
            state["model"] = res
            return len(preds)

        def _monitor():
            df = state["clean"]
            test_ids = set(state["model"].predictions["record_id"])
            is_test = df["record_id"].isin(test_ids)
            feats = [*spec.numeric_features, *spec.categorical_features]
            table = monitoring.psi_table(df[~is_test], df[is_test], feats)
            db.write_df(table, "feature_psi")
            return len(table)

        def _export():
            out = settings.processed_dir(spec.id)
            state["clean"].to_csv(out / "clean_loans.csv", index=False)
            state["clean"].to_parquet(out / "clean_loans.parquet", index=False)
            m = state["model"]
            summary = {
                "run_id": run_id, "dataset": spec.id, "dataset_sha256": state["sha"],
                "rows_raw": len(state["raw"]), "rows_clean": len(state["clean"]),
                "quality_raw": state["score_raw"], "quality_clean": state["score_clean"],
                "champion": m.champion, "champion_id": m.champion_id,
                "metrics": m.metrics.drop(columns=["label"]).to_dict(orient="records"),
            }
            (settings.reports_dir(spec.id) / f"run_{run_id}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            return 1

        for i, fn in enumerate([_ingest, _validate_raw, _clean, _validate_clean, _features, _model, _monitor, _export]):
            step(i, fn)

        champ = state["model"].metrics.query("is_champion == 1").iloc[0]
        duration = time.perf_counter() - started
        db.execute(
            """UPDATE pipeline_runs SET finished_at=:f, status='sucesso', duration_s=:d, dataset_sha256=:sha,
               rows_raw=:rr, rows_clean=:rc, quality_raw=:qr, quality_score=:q, champion_model=:cm, champion_auc=:ca WHERE run_id=:r""",
            f=_now(), d=duration, sha=state["sha"], rr=len(state["raw"]), rc=len(state["clean"]),
            qr=state["score_raw"], q=state["score_clean"], cm=state["model"].champion_id, ca=float(champ["roc_auc"]), r=run_id)
        status = "sucesso"
    except Exception as exc:  # registra a falha para o painel de monitoramento
        log.exception("Pipeline falhou")
        db.execute("UPDATE pipeline_runs SET finished_at=:f, status='falha', error=:e, duration_s=:d WHERE run_id=:r",
                   f=_now(), e=str(exc)[:500], d=time.perf_counter() - started, r=run_id)
        raise
    finally:
        if timings:
            db.write_df(pd.DataFrame([{"run_id": run_id, "step": s, "seconds": t, "rows_out": n}
                                      for s, t, n in timings]), "step_timings", if_exists="append")
    if progress:
        progress("Concluído", len(STEPS), len(STEPS))
    return {"run_id": run_id, "dataset": spec.id, "status": status, "duration_s": round(time.perf_counter() - started, 2)}


def has_successful_run(dataset: str | None = None) -> bool:
    if dataset is not None:
        with context.use(dataset):
            return has_successful_run()
    if not db.table_exists("clean_loans"):
        return False
    df = db.read_sql("SELECT COUNT(*) AS n FROM pipeline_runs WHERE status='sucesso'")
    return int(df["n"].iloc[0]) > 0


def main() -> None:
    from ..datasets import DATASETS

    parser = argparse.ArgumentParser(description="Executa a pipeline Vértice Credit Analytics")
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all", help="base a processar")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    for ds in (DATASETS if args.dataset == "all" else [args.dataset]):
        print(json.dumps(run(dataset=ds), ensure_ascii=False))


if __name__ == "__main__":
    main()
