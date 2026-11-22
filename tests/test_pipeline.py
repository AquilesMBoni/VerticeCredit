import numpy as np
import pandas as pd
import pytest

from vertice import context
from vertice.datasets import br_rules as R
from vertice.pipeline import cleaning, features, monitoring, validation


# ----------------------------------------------------------------- base EUA
def test_ingestion_keeps_raw_layer_intact(raw_us):
    assert len(raw_us) == 32581
    assert raw_us["record_id"].is_unique
    assert raw_us["loan_int_rate"].isna().sum() > 0  # bruto não é alterado


@pytest.mark.parametrize("ds", ["us", "br"])
def test_raw_fails_and_clean_passes_contract(ds, raw_us, raw_br):
    raw = {"us": raw_us, "br": raw_br}[ds]
    with context.use(ds):
        rep_raw, s_raw = validation.run_quality_checks(raw.drop(columns="record_id"), "bruto")
        clean, log = cleaning.clean(raw)
        rep_clean, s_clean = validation.run_quality_checks(clean, "limpo")
    assert (rep_raw.status == "reprovado").sum() >= 5 and s_raw < 90
    assert (rep_clean.status == "aprovado").all(), rep_clean[rep_clean.status != "aprovado"]
    assert s_clean == 100.0
    assert len(log) > 0


def test_us_cleaning_rules(raw_us):
    with context.use("us"):
        clean, log = cleaning.clean(raw_us)
    assert clean["person_age"].between(18, 100).all()
    assert clean.drop(columns="record_id").duplicated().sum() == 0
    assert clean[["loan_int_rate", "person_emp_length"]].notna().all().all()
    assert (clean["person_emp_length"] <= clean["person_age"] - 14).all()


def test_us_rate_imputation_uses_grade_median():
    from vertice.datasets import us
    from vertice.datasets.base import CleaningLog
    df = pd.DataFrame({"loan_grade": ["A", "A", "A", "G"], "loan_int_rate": [7.0, 9.0, np.nan, 20.0]})
    out = us._impute_rate(df.copy(), CleaningLog())
    assert out.loc[2, "loan_int_rate"] == 8.0
    assert out["loan_int_rate_imputed"].tolist() == [0, 0, 1, 0]


# ----------------------------------------------------------------- base Brasil
def test_br_parsers():
    from vertice.datasets.br import parse_brl, parse_date
    s = parse_brl(pd.Series(["R$ 3.450,00", "1234.50", "", None, "R$ 0,00"]))
    assert s.iloc[0] == 3450.0 and s.iloc[1] == 1234.5 and s.iloc[2:4].isna().all() and s.iloc[4] == 0
    d = parse_date(pd.Series(["05/03/2026", "2026-03-05", "31/02/2026"]))
    assert d.iloc[0] == d.iloc[1] and pd.isna(d.iloc[2])


def test_br_price_table():
    # R$ 10.000 em 12x a 2% a.m. → parcela de R$ 945,60 (Tabela Price)
    assert round(float(R.parcela_price(10_000, 2.0, 12)), 2) == 945.60


def test_br_clean_respects_brazilian_rules(raw_br):
    with context.use("br"):
        clean, log = cleaning.clean(raw_br)
        df = features.add_features(clean)
    inss = df[df["modalidade"] == "Consignado INSS"]
    assert (inss["taxa_juros_am"] <= R.TETO_JUROS_INSS_AM).all()
    assert (inss["ocupacao"] == "Aposentado/Pensionista INSS").all()
    margem = df["modalidade"].map(R.MARGEM_CONSIGNAVEL)
    assert (df.loc[margem.notna(), "comprometimento_renda"] <= margem.dropna() + 0.005).all()
    assert set(df["uf"]) <= set(R.UFS) and set(df["possui_restricao"]) == {"Sim", "Não"}
    assert df["renda_mensal"].notna().all() and pd.to_datetime(df["data_contratacao"]).max() <= pd.Timestamp("2026-06-30")
    assert {"Acima do teto do INSS", "Margem consignável estourada", "Sem elegibilidade"} <= set(log["step"])


def test_information_value_ranks_strong_signals(raw_us):
    with context.use("us"):
        clean, _ = cleaning.clean(raw_us)
        _, iv = features.woe_iv(features.add_features(clean))
    top = iv.head(5)["feature"].tolist()
    assert "loan_grade" in top and "loan_percent_income" in top


def test_psi_is_zero_for_identical_and_high_for_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    assert monitoring.psi(a, a) < 1e-6
    assert monitoring.psi(a, a + 1.5) > 0.25


# ----------------------------------------------------------------- pipeline completa
@pytest.mark.parametrize("ds", ["us", "br"])
def test_full_pipeline_registers_champion(ds, pipeline_run):
    from vertice import services
    assert pipeline_run[ds]["status"] == "sucesso"
    with context.use(ds):
        champ = services.champion_record()
        metrics = services.table("model_metrics").set_index("model")
        pol = services.policy_curve(lgd=0.6)
    assert champ["metrics"]["roc_auc"] > 0.8
    assert metrics.loc["baseline", "roc_auc"] == 0.5
    assert pol["approval_rate"].is_monotonic_increasing and pol["approval_rate"].iloc[-1] == 1.0


def test_grade_filter_recomputes_metrics(pipeline_run):
    from vertice import services
    with context.use("br"):
        preds = services.table("test_predictions")
        sub = services.filter_grades(preds, ["AA", "A"])
        assert set(sub["rating"]) <= {"AA", "A"}
        assert services.overview(["G", "H"])["default_rate"] > services.overview(["AA"])["default_rate"]
