"""Base EUA: Credit Risk Dataset (Kaggle, CC0).

Dados que simulam um bureau de crédito, com valores em dólar. A página do
Kaggle não informa país, período nem método de geração.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa

from .base import DatasetSpec, FormField, Rule

AGE_MIN, AGE_MAX, EMP_MAX = 18, 100, 60
HOME = ["RENT", "OWN", "MORTGAGE", "OTHER"]
INTENT = ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
GRADES = list("ABCDEFG")

COLUMN_DOCS = {
    "person_age": "Idade do solicitante (anos)",
    "person_income": "Renda anual (US$)",
    "person_home_ownership": "Situação de moradia (RENT, OWN, MORTGAGE, OTHER)",
    "person_emp_length": "Tempo de emprego (anos)",
    "loan_intent": "Finalidade do empréstimo",
    "loan_grade": "Nota de risco atribuída pelo credor (A = melhor, G = pior)",
    "loan_amnt": "Valor solicitado (US$)",
    "loan_int_rate": "Taxa de juros anual (%)",
    "loan_status": "Alvo: 1 = inadimplente (default), 0 = adimplente",
    "loan_percent_income": "Valor do empréstimo ÷ renda anual",
    "cb_person_default_on_file": "Possui default anterior no bureau (Y/N)",
    "cb_person_cred_hist_length": "Tempo de histórico de crédito (anos)",
}

RULES = {
    "not_nullable": Rule("Completude", "Valor obrigatório ausente", "alta"),
    "age_range": Rule("Validade", f"Idade entre {AGE_MIN} e {AGE_MAX} anos", "alta", "person_age"),
    "income_positive": Rule("Validade", "Renda anual maior que zero", "alta", "person_income"),
    "home_domain": Rule("Validade", "Moradia dentro do domínio conhecido", "média", "person_home_ownership"),
    "emp_range": Rule("Validade", f"Tempo de emprego entre 0 e {EMP_MAX} anos", "alta", "person_emp_length"),
    "intent_domain": Rule("Validade", "Finalidade dentro do domínio conhecido", "média", "loan_intent"),
    "grade_domain": Rule("Validade", "Nota de crédito entre A e G", "média", "loan_grade"),
    "amount_positive": Rule("Validade", "Valor do empréstimo maior que zero", "alta", "loan_amnt"),
    "rate_range": Rule("Validade", "Taxa de juros entre 0% e 40% a.a.", "média", "loan_int_rate"),
    "target_binary": Rule("Validade", "Alvo binário (0 ou 1)", "alta", "loan_status"),
    "pct_income_range": Rule("Validade", "Comprometimento de renda entre 0 e 1", "média", "loan_percent_income"),
    "default_flag_domain": Rule("Validade", "Default anterior é Y ou N", "média", "cb_person_default_on_file"),
    "hist_range": Rule("Validade", "Histórico de crédito entre 0 e 60 anos", "média", "cb_person_cred_hist_length"),
    "no_duplicates": Rule("Unicidade", "Registro não duplicado", "média", "(linha)"),
    "emp_vs_age": Rule("Consistência", "Tempo de emprego compatível com a idade (início aos 14)", "alta", "person_emp_length"),
    "pct_income_consistent": Rule("Consistência", "Comprometimento = valor ÷ renda (tolerância 0,01)", "baixa", "loan_percent_income"),
}


def build_schema() -> pa.DataFrameSchema:
    c = pa.Check
    return pa.DataFrameSchema(
        {
            "person_age": pa.Column(float, c.in_range(AGE_MIN, AGE_MAX, error="age_range"), coerce=True),
            "person_income": pa.Column(float, c.gt(0, error="income_positive"), coerce=True),
            "person_home_ownership": pa.Column(str, c.isin(HOME, error="home_domain")),
            "person_emp_length": pa.Column(float, c.in_range(0, EMP_MAX, error="emp_range"), nullable=False, coerce=True),
            "loan_intent": pa.Column(str, c.isin(INTENT, error="intent_domain")),
            "loan_grade": pa.Column(str, c.isin(GRADES, error="grade_domain")),
            "loan_amnt": pa.Column(float, c.gt(0, error="amount_positive"), coerce=True),
            "loan_int_rate": pa.Column(float, c.in_range(0, 40, error="rate_range"), nullable=False, coerce=True),
            "loan_status": pa.Column(int, c.isin([0, 1], error="target_binary"), coerce=True),
            "loan_percent_income": pa.Column(float, c.in_range(0, 1, error="pct_income_range"), coerce=True),
            "cb_person_default_on_file": pa.Column(str, c.isin(["Y", "N"], error="default_flag_domain")),
            "cb_person_cred_hist_length": pa.Column(float, c.in_range(0, 60, error="hist_range"), coerce=True),
        },
        checks=[
            pa.Check(lambda d: ~d.drop(columns=["record_id"], errors="ignore").duplicated(), error="no_duplicates"),
            pa.Check(lambda d: d["person_emp_length"].isna() | (d["person_emp_length"] <= d["person_age"] - 14), error="emp_vs_age"),
            pa.Check(lambda d: (d["loan_percent_income"] - d["loan_amnt"] / d["person_income"]).abs() <= 0.01,
                     error="pct_income_consistent"),
        ],
        strict=False,
    )


# ---------------------------------------------------------------- limpeza
def _standardize(df, lg):
    changed = 0
    for col in ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]:
        norm = df[col].astype(str).str.strip().str.upper()
        changed += int((norm != df[col].astype(str)).sum())
        df[col] = norm
    lg.add("Padronização", "Texto categórico em caixa alta e sem espaços", "Normalizar", len(df), len(df), changed)
    return df


def _drop_duplicates(df, lg):
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "record_id"], keep="first")
    lg.add("Duplicatas", "Registros idênticos em todas as colunas", "Remover", before, len(df), before - len(df))
    return df


def _invalid_age(df, lg):
    before = len(df)
    df = df[df["person_age"].between(AGE_MIN, AGE_MAX)]
    lg.add("Idade impossível", f"Idade fora de {AGE_MIN}–{AGE_MAX} anos (ex.: 144)", "Remover", before, len(df), before - len(df))
    return df


def _invalid_emp_length(df, lg):
    bad = (df["person_emp_length"] > EMP_MAX) | (df["person_emp_length"] > df["person_age"] - 14)
    df.loc[bad, "person_emp_length"] = np.nan
    lg.add("Tempo de emprego inválido", "Maior que 60 anos ou incompatível com a idade (ex.: 123)",
           "Anular para imputar", len(df), len(df), int(bad.sum()))
    return df


def _winsorize_income(df, lg):
    cap = float(df["person_income"].quantile(0.995))
    flag = df["person_income"] > cap
    df["income_capped"] = flag.astype(int)
    df["person_income"] = df["person_income"].clip(upper=cap)
    lg.add("Renda extrema", f"Renda acima do p99,5 (US$ {cap:,.0f}) limitada ao teto".replace(",", "."),
           "Winsorizar", len(df), len(df), int(flag.sum()))
    return df


def _recompute_pct_income(df, lg):
    exact = (df["loan_amnt"] / df["person_income"]).round(4)
    diff = (exact - df["loan_percent_income"]).abs() > 0.01
    df["loan_percent_income"] = exact.clip(upper=1.0)
    lg.add("Comprometimento de renda", "Recalcular valor ÷ renda com 4 casas (original arredondado)",
           "Recalcular", len(df), len(df), int(diff.sum()))
    return df


def _impute_rate(df, lg):
    missing = df["loan_int_rate"].isna()
    df["loan_int_rate_imputed"] = missing.astype(int)
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df.groupby("loan_grade")["loan_int_rate"].transform("median")).round(2)
    lg.add("Taxa de juros ausente", "Mediana da taxa dentro da mesma nota (A–G), que define a taxa",
           "Imputar", len(df), len(df), int(missing.sum()))
    return df


def _impute_emp_length(df, lg):
    missing = df["person_emp_length"].isna()
    df["person_emp_length_imputed"] = missing.astype(int)
    bands = pd.cut(df["person_age"], [0, 25, 35, 45, 60, 200])
    df["person_emp_length"] = df["person_emp_length"].fillna(
        df.groupby(bands, observed=True)["person_emp_length"].transform("median"))
    lg.add("Tempo de emprego ausente", "Mediana dentro da mesma faixa etária", "Imputar", len(df), len(df), int(missing.sum()))
    return df


def _cast_types(df, lg):
    for col in ("person_age", "person_income", "loan_amnt", "loan_status", "cb_person_cred_hist_length"):
        df[col] = df[col].round().astype("int64")
    df["person_emp_length"] = df["person_emp_length"].astype(float)
    lg.add("Tipagem", "Inteiros para contagens e valores monetários inteiros", "Converter", len(df), len(df), 0)
    return df


STEPS = (_standardize, _drop_duplicates, _invalid_age, _invalid_emp_length, _winsorize_income,
         _recompute_pct_income, _impute_rate, _impute_emp_length, _cast_types)


# ---------------------------------------------------------------- atributos
AGE_BINS, AGE_LABELS = [17, 25, 35, 45, 60, 200], ["18–25", "26–35", "36–45", "46–60", "60+"]
INCOME_BINS = [0, 25_000, 50_000, 75_000, 100_000, 150_000, np.inf]
INCOME_LABELS = ["até 25k", "25–50k", "50–75k", "75–100k", "100–150k", "150k+"]

FEATURE_DOCS = {
    "interest_burden": "Juros anuais do empréstimo ÷ renda anual",
    "emp_to_age": "Anos de emprego ÷ anos em idade ativa (estabilidade)",
    "credit_hist_to_age": "Anos de histórico de crédito ÷ idade adulta",
    "age_band": "Faixa etária",
    "income_band": "Faixa de renda anual",
    "grade_num": "Nota A–G convertida em 1–7",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["interest_burden"] = (out["loan_amnt"] * out["loan_int_rate"] / 100 / out["person_income"]).round(5)
    out["emp_to_age"] = (out["person_emp_length"] / (out["person_age"] - 14).clip(lower=1)).clip(0, 1).round(4)
    out["credit_hist_to_age"] = (out["cb_person_cred_hist_length"] / (out["person_age"] - 17).clip(lower=1)).clip(0, 2).round(4)
    out["age_band"] = pd.cut(out["person_age"], AGE_BINS, labels=AGE_LABELS).astype(str)
    out["income_band"] = pd.cut(out["person_income"], INCOME_BINS, labels=INCOME_LABELS, right=False).astype(str)
    out["grade_num"] = out["loan_grade"].map({g: i for i, g in enumerate(GRADES, start=1)})
    return out


def revenue_and_loss(preds: pd.DataFrame, lgd: float):
    amt = preds["loan_amnt"].to_numpy()
    return amt * preds["loan_int_rate"].to_numpy() / 100, amt * lgd


def prepare_payload(payload: dict) -> pd.DataFrame:
    base = pd.DataFrame([payload])
    base["loan_percent_income"] = (base["loan_amnt"] / base["person_income"]).clip(upper=1.0)
    return add_features(base)


def insights(clean: pd.DataFrame, policy: pd.DataFrame) -> list[tuple[str, str]]:
    from ._fmt import pct

    out = []
    g = clean.groupby("loan_grade")["loan_status"].mean()
    if len(g) > 1:
        lo, hi = g.index.min(), g.index.max()
        out.append(("fa-ranking-star", f"A nota ordena bem o risco: o default sai de <b>{pct(g[lo])}</b> na nota {lo} "
                                       f"para <b>{pct(g[hi])}</b> na nota {hi}."))
    hi_ = clean["loan_percent_income"] > 0.30
    if hi_.any() and (~hi_).any():
        out.append(("fa-percent", f"Quem compromete mais de 30% da renda inadimple <b>{pct(clean.loc[hi_, 'loan_status'].mean())}</b> "
                                  f"das vezes, contra <b>{pct(clean.loc[~hi_, 'loan_status'].mean())}</b> no restante."))
    home = clean.groupby("person_home_ownership")["loan_status"].mean()
    if {"RENT", "MORTGAGE"} <= set(home.index):
        out.append(("fa-house", f"Inquilinos (RENT) têm default de <b>{pct(home['RENT'])}</b>; quem financia a casa "
                                f"(MORTGAGE), <b>{pct(home['MORTGAGE'])}</b>."))
    prior = clean.groupby("cb_person_default_on_file")["loan_status"].mean()
    if {"Y", "N"} <= set(prior.index):
        out.append(("fa-clock-rotate-left", f"Ter default anterior no bureau eleva a taxa de <b>{pct(prior['N'])}</b> "
                                            f"para <b>{pct(prior['Y'])}</b>."))
    return out


SPEC = DatasetSpec(
    id="us", name="Credit Risk Dataset (Kaggle)", short="EUA", currency="US$",
    raw_file="credit_risk_dataset.csv", is_synthetic=True,
    source_url="https://www.kaggle.com/datasets/laotse/credit-risk-dataset",
    source_label="kaggle.com/datasets/laotse/credit-risk-dataset", license="CC0 1.0 (domínio público)",
    source_html=("<b>Credit Risk Dataset</b>, publicado por Lao Tse no Kaggle. A descrição oficial diz apenas que as "
                 "colunas <b>simulam dados de bureau de crédito</b>: não informa país, período nem como os dados foram "
                 "gerados. Os valores estão em dólar e as categorias (moradia, notas A–G) seguem o mercado americano."),
    read_raw=lambda path: pd.read_csv(path),
    expected_columns=tuple(COLUMN_DOCS), column_docs=COLUMN_DOCS,
    target="loan_status", grade="loan_grade", grade_order=tuple(GRADES),
    amount="loan_amnt", income="person_income", income_label="Renda anual (US$)",
    rate="loan_int_rate", rate_label="Taxa de juros (% a.a.)",
    purpose="loan_intent", purpose_label="Finalidade", age="person_age",
    build_schema=build_schema, rules=RULES, validation_view=lambda df: df,
    cleaning_steps=STEPS, add_features=add_features, feature_docs=FEATURE_DOCS,
    numeric_features=("person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate",
                      "loan_percent_income", "cb_person_cred_hist_length", "interest_burden", "emp_to_age",
                      "credit_hist_to_age"),
    categorical_features=("person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"),
    imputed={"loan_int_rate_imputed": ("loan_int_rate",), "person_emp_length_imputed": ("person_emp_length", "emp_to_age")},
    policy_cols=("loan_amnt", "loan_int_rate", "loan_grade"),
    revenue_and_loss=revenue_and_loss, default_cutoff=0.30,
    form_fields=(
        FormField("person_age", "Idade", "int", 29, 18, 100),
        FormField("person_income", "Renda anual (US$)", "int", 52_000, 1000, 2_000_000, 1000),
        FormField("person_emp_length", "Anos de emprego", "int", 4, 0, 60),
        FormField("person_home_ownership", "Moradia", "select", "RENT", options=tuple(HOME)),
        FormField("loan_amnt", "Valor solicitado (US$)", "int", 9_000, 500, 100_000, 500),
        FormField("loan_intent", "Finalidade", "select", "EDUCATION", options=tuple(INTENT)),
        FormField("loan_grade", "Nota de crédito", "select", "B", options=tuple(GRADES)),
        FormField("loan_int_rate", "Taxa de juros (% a.a.)", "float", 11.5, 5.0, 30.0, 0.25),
        FormField("cb_person_default_on_file", "Default anterior", "select", "N", options=("N", "Y"),
                  labels={"N": "Não", "Y": "Sim"}),
        FormField("cb_person_cred_hist_length", "Anos de histórico de crédito", "int", 5, 0, 60),
    ),
    prepare_payload=prepare_payload,
    example_payload={"person_age": 29, "person_income": 52000, "person_home_ownership": "RENT", "person_emp_length": 4,
                     "loan_intent": "EDUCATION", "loan_grade": "B", "loan_amnt": 9000, "loan_int_rate": 11.5,
                     "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 5},
    predict_route="/predict",
    segment_dims={"loan_grade": "Nota de crédito", "loan_intent": "Finalidade", "person_home_ownership": "Moradia",
                  "age_band": "Faixa etária", "income_band": "Faixa de renda", "cb_person_default_on_file": "Default anterior"},
    dim_orders={"age_band": AGE_LABELS, "income_band": INCOME_LABELS, "loan_grade": GRADES},
    dist_vars={"loan_percent_income": "Comprometimento de renda", "loan_int_rate": "Taxa de juros (% a.a.)",
               "person_income": "Renda anual (US$)", "loan_amnt": "Valor solicitado (US$)",
               "interest_burden": "Peso dos juros na renda", "person_age": "Idade"},
    corr_cols={"loan_status": "default", "loan_percent_income": "% renda", "loan_int_rate": "taxa", "grade_num": "nota",
               "person_income": "renda", "loan_amnt": "valor", "person_emp_length": "emprego", "person_age": "idade",
               "cb_person_cred_hist_length": "histórico"},
    impossible_vars=(("person_age", "Idade (anos)"), ("person_emp_length", "Tempo de emprego (anos)")),
    compare_vars={"person_income": "Renda anual (US$)", "person_age": "Idade (anos)",
                  "person_emp_length": "Tempo de emprego (anos)", "loan_int_rate": "Taxa de juros (% a.a.)",
                  "loan_percent_income": "Comprometimento de renda"},
    impute_view=("loan_int_rate", "loan_int_rate_imputed", "Imputação da taxa de juros por nota", "Taxa (% a.a.)"),
    scatter=dict(x="person_income", y="loan_amnt", xlabel="Renda anual (US$)", ylabel="Valor solicitado (US$)",
                 ratios=(0.1, 0.3, 0.5), ratio_label="{p}% da renda", ymax=36000,
                 note="Acima da linha de 30% da renda, os pontos laranja dominam: o comprometimento de renda é o sinal "
                      "mais visível de risco na base."),
    insights=insights,
    texts={
        "ingestion_note": "Repare na primeira linha: <b>123 anos de emprego</b> para alguém de 22 anos. Esse tipo de erro "
                          "é o que o contrato de qualidade da próxima fase existe para pegar.",
        "quality_note": "Idades de 123 e 144 anos e 123 anos de emprego são erros de digitação, não clientes raros. "
                        "A limpeza remove as idades impossíveis e anula os tempos de emprego inválidos para imputação.",
        "iv_note": "IV acima de 0,5 costuma indicar vazamento de dados e pede verificação. Aqui não é: <b>loan_grade</b> e "
                   "<b>loan_int_rate</b> são definidos pelo próprio credor a partir do risco percebido. Em produção, eles "
                   "só podem entrar se forem conhecidos <i>antes</i> da decisão de crédito.",
        "cleaning_intro": "Regras aplicadas em ordem, todas registradas no <code>cleaning_log</code>.",
        "policy_intro": "Conta simples de um ano: juros anuais recebidos dos bons pagadores menos a perda (LGD × valor) "
                        "dos inadimplentes aprovados.",
    },
)
