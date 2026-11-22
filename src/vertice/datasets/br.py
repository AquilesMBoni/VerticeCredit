"""Base Brasil: crédito pessoa física com regras brasileiras (sintética, calibrada).

Veja `br_generator.py` para a origem e `br_rules.py` para as regras e fontes.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
import pandera.pandas as pa

from . import br_rules as R
from ._fmt import num, pct
from .base import DatasetSpec, FormField, Rule

DATA_CORTE = pd.Timestamp("2026-06-30")  # fim da janela de contratação; desfecho observado em 12 meses
BRL_RE = re.compile(r"^R\$ \d{1,3}(\.\d{3})*,\d{2}$")
DATA_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")

COLUMN_DOCS = {
    "id_proposta": "Identificador da proposta",
    "data_contratacao": "Data da contratação (dd/mm/aaaa)",
    "uf": "Unidade da Federação do cliente",
    "idade": "Idade do cliente (anos)",
    "ocupacao": "Vínculo de renda: CLT, servidor, aposentado INSS, autônomo ou MEI",
    "renda_mensal": "Renda mensal bruta, em texto no formato brasileiro (R$ 1.234,56)",
    "tempo_emprego_meses": "Tempo no emprego atual (meses); vazio para aposentados",
    "score_bureau": "Score de bureau (0–1000); vazio quando o cliente não tem histórico",
    "possui_restricao": "Nome com restrição (negativado) em bureau: Sim/Não",
    "modalidade": "Modalidade da operação (consignados, pessoal, veículos)",
    "valor_contratado": "Valor liberado (R$)",
    "prazo_meses": "Número de parcelas mensais",
    "taxa_juros_am": "Taxa de juros contratada (% ao mês)",
    "rating": "Classificação de risco na concessão (AA a H)",
    "inadimplente": "Alvo: 1 = atraso acima de 90 dias nos 12 primeiros meses",
}

RULES = {
    "not_nullable": Rule("Completude", "Valor obrigatório ausente", "alta"),
    "idade_range": Rule("Validade", "Idade entre 18 e 100 anos", "alta", "idade"),
    "uf_domain": Rule("Validade", "UF é uma das 27 siglas oficiais", "média", "uf"),
    "ocupacao_domain": Rule("Validade", "Ocupação dentro do domínio conhecido", "média", "ocupacao"),
    "modalidade_domain": Rule("Validade", "Modalidade dentro do domínio conhecido", "média", "modalidade"),
    "rating_domain": Rule("Validade", "Rating entre AA e H", "média", "rating"),
    "restricao_domain": Rule("Validade", "Restrição é Sim ou Não", "baixa", "possui_restricao"),
    "score_range": Rule("Validade", "Score entre 0 e 1000", "média", "score_bureau"),
    "valor_positive": Rule("Validade", "Valor contratado maior que zero", "alta", "valor_contratado"),
    "target_binary": Rule("Validade", "Alvo binário (0 ou 1)", "alta", "inadimplente"),
    "formato_moeda": Rule("Formato", "Renda no formato brasileiro (R$ 1.234,56)", "baixa", "renda_mensal"),
    "formato_data": Rule("Formato", "Data válida no formato dd/mm/aaaa", "média", "data_contratacao"),
    "no_duplicates": Rule("Unicidade", "Proposta não duplicada", "média", "(linha)"),
    "data_futura": Rule("Consistência", "Contratação até a data de corte (30/06/2026)", "alta", "data_contratacao"),
    "emprego_vs_idade": Rule("Consistência", "Tempo de emprego compatível com a idade", "média", "tempo_emprego_meses"),
    "teto_inss": Rule("Conformidade regulatória", f"Consignado INSS com juros até {num(R.TETO_JUROS_INSS_AM)}% a.m. (teto CNPS)",
                      "alta", "taxa_juros_am"),
    "margem_consignavel": Rule("Conformidade regulatória", "Parcela do consignado dentro da margem (45% INSS/servidor, 35% CLT)",
                               "alta", "valor_contratado"),
    "elegibilidade": Rule("Conformidade regulatória", "Consignado só para quem tem o vínculo exigido", "alta", "modalidade"),
    "prazo_maximo": Rule("Conformidade regulatória", "Prazo dentro do máximo da modalidade (INSS: 108 meses)", "média", "prazo_meses"),
    "piso_renda": Rule("Conformidade regulatória", f"Renda formal ≥ salário mínimo (R$ {num(R.SALARIO_MINIMO)})", "baixa",
                       "renda_mensal"),
}


# ---------------------------------------------------------------- leitura e visão de validação
def read_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig",
                       dtype={"renda_mensal": str, "data_contratacao": str, "uf": str, "rating": str,
                              "possui_restricao": str, "ocupacao": str})


def parse_brl(s: pd.Series) -> pd.Series:
    """'R$ 3.450,00' → 3450.0; aceita também '3450.00'. Texto vazio vira nulo."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    txt = s.astype("string").str.replace("R$", "", regex=False).str.strip()
    br = txt.str.contains(",", regex=False).fillna(False).astype(bool)
    br_num = pd.to_numeric(txt.str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
    plain = pd.to_numeric(txt, errors="coerce")
    return pd.Series(np.where(br, br_num.astype(float), plain.astype(float)), index=s.index, dtype=float)


def parse_date(s: pd.Series) -> pd.Series:
    d1 = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    d2 = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    return d1.fillna(d2)


def validation_view(df: pd.DataFrame) -> pd.DataFrame:
    """Converte só o necessário para checar as regras, sem corrigir nada."""
    v = df.copy()
    raw_text = not pd.api.types.is_numeric_dtype(df["renda_mensal"])
    if raw_text:
        txt = df["renda_mensal"].astype("string").str.strip()
        v["_fmt_moeda_ok"] = (txt.isna() | txt.eq("") | txt.str.match(BRL_RE.pattern)).fillna(True).astype(bool)
        dtxt = df["data_contratacao"].astype("string")
        parsed_br = pd.to_datetime(dtxt, format="%d/%m/%Y", errors="coerce")
        v["_fmt_data_ok"] = (dtxt.str.match(DATA_RE.pattern).fillna(False).astype(bool) & parsed_br.notna()).to_numpy()
    else:
        v["_fmt_moeda_ok"] = True
        v["_fmt_data_ok"] = True
    v["renda_mensal"] = parse_brl(df["renda_mensal"])
    v["_data"] = parse_date(df["data_contratacao"].astype("string"))
    v["_parcela"] = R.parcela_price(v["valor_contratado"], v["taxa_juros_am"], v["prazo_meses"])
    v["_compr"] = v["_parcela"] / v["renda_mensal"]
    return v


def build_schema() -> pa.DataFrameSchema:
    c = pa.Check
    margem = lambda d: d["modalidade"].map(R.MARGEM_CONSIGNAVEL)  # noqa: E731
    elig = lambda d: [o in R.ELEGIBILIDADE.get(m, {o}) for m, o in zip(d["modalidade"], d["ocupacao"])]  # noqa: E731
    prazo_max = lambda d: d["modalidade"].map(R.PRAZO_MAXIMO)  # noqa: E731
    return pa.DataFrameSchema(
        {
            "idade": pa.Column(float, c.in_range(18, 100, error="idade_range"), coerce=True),
            "uf": pa.Column(str, c.isin(list(R.UFS), error="uf_domain")),
            "ocupacao": pa.Column(str, c.isin(R.OCUPACOES, error="ocupacao_domain")),
            "modalidade": pa.Column(str, c.isin(R.MODALIDADES, error="modalidade_domain")),
            "rating": pa.Column(str, c.isin(R.RATINGS, error="rating_domain")),
            "possui_restricao": pa.Column(str, c.isin(["Sim", "Não"], error="restricao_domain")),
            "renda_mensal": pa.Column(float, nullable=False, coerce=True),
            "score_bureau": pa.Column(float, c.in_range(0, 1000, error="score_range"), nullable=False, coerce=True),
            "tempo_emprego_meses": pa.Column(float, nullable=True, coerce=True),
            "valor_contratado": pa.Column(float, c.gt(0, error="valor_positive"), coerce=True),
            "inadimplente": pa.Column(int, c.isin([0, 1], error="target_binary"), coerce=True),
        },
        checks=[
            pa.Check(lambda d: ~d[[k for k in COLUMN_DOCS if k in d]].duplicated(), error="no_duplicates"),
            pa.Check(lambda d: d["_fmt_moeda_ok"].astype(bool), error="formato_moeda"),
            pa.Check(lambda d: d["_fmt_data_ok"].astype(bool), error="formato_data"),
            pa.Check(lambda d: d["_data"].isna() | (d["_data"] <= DATA_CORTE), error="data_futura"),
            pa.Check(lambda d: d["tempo_emprego_meses"].isna() | (d["tempo_emprego_meses"] <= (d["idade"] - 14) * 12),
                     error="emprego_vs_idade"),
            pa.Check(lambda d: (d["modalidade"] != "Consignado INSS") | (d["taxa_juros_am"] <= R.TETO_JUROS_INSS_AM + 1e-9),
                     error="teto_inss"),
            pa.Check(lambda d: margem(d).isna() | d["_compr"].isna() | (d["_compr"] <= margem(d) + 0.005), error="margem_consignavel"),
            pa.Check(lambda d: pd.Series(elig(d), index=d.index), error="elegibilidade"),
            pa.Check(lambda d: prazo_max(d).isna() | (d["prazo_meses"] <= prazo_max(d)), error="prazo_maximo"),
            pa.Check(lambda d: ~d["ocupacao"].isin(R.RENDA_FORMAL) | d["renda_mensal"].isna()
                     | (d["renda_mensal"] >= R.SALARIO_MINIMO - 0.01), error="piso_renda"),
        ],
        strict=False,
    )


# ---------------------------------------------------------------- limpeza
def _key(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().casefold()


UF_BY_NAME = {_key(nome): sigla for sigla, (nome, _, _) in R.UFS.items()}
OCUP_BY_KEY = {_key(o): o for o in R.OCUPACOES}


def _padronizar(df, lg):
    before = df[["uf", "ocupacao", "rating", "possui_restricao"]].astype(str).copy()
    uf = df["uf"].astype(str).str.strip().str.upper()
    df["uf"] = uf.where(uf.isin(list(R.UFS)), df["uf"].map(lambda v: UF_BY_NAME.get(_key(v), v)))
    df["ocupacao"] = df["ocupacao"].map(lambda v: OCUP_BY_KEY.get(_key(v), v))
    df["rating"] = df["rating"].astype(str).str.strip().str.upper()
    df["possui_restricao"] = df["possui_restricao"].map(
        lambda v: "Sim" if _key(v) in ("s", "sim") else "Não" if _key(v) in ("n", "nao") else v)
    changed = int((before != df[before.columns].astype(str)).any(axis=1).sum())
    lg.add("Padronização de texto", "UF em sigla (inclusive nomes por extenso), ocupação canônica, rating em caixa alta, "
           "restrição Sim/Não", "Normalizar", len(df), len(df), changed)
    return df


def _duplicatas(df, lg):
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "record_id"], keep="first")
    lg.add("Duplicatas", "Propostas idênticas em todas as colunas", "Remover", before, len(df), before - len(df))
    return df


def _moeda(df, lg):
    txt = df["renda_mensal"].astype("string").str.strip()
    fora = ~(txt.isna() | txt.eq("") | txt.str.match(BRL_RE.pattern)).fillna(True).astype(bool)
    val = parse_brl(df["renda_mensal"])
    zero = val.eq(0)
    df["renda_mensal"] = val.mask(zero)
    lg.add("Moeda em texto", "Converter 'R$ 1.234,56' (e '1234.56') em número; R$ 0,00 vira ausente",
           "Converter", len(df), len(df), int(fora.sum() + zero.sum()))
    return df


def _datas(df, lg):
    before = len(df)
    d = parse_date(df["data_contratacao"].astype("string"))
    invalid = d.isna()
    df = df[~invalid].copy()
    df["data_contratacao"] = d[~invalid]
    lg.add("Data inválida", "Datas impossíveis (ex.: 31/02) ou fora dos formatos dd/mm/aaaa e aaaa-mm-dd", "Remover",
           before, len(df), int(invalid.sum()))
    before = len(df)
    futura = df["data_contratacao"] > DATA_CORTE
    df = df[~futura]
    lg.add("Data futura", "Contratação depois da data de corte (30/06/2026)", "Remover", before, len(df), int(futura.sum()))
    return df


def _idade(df, lg):
    before = len(df)
    df = df[df["idade"].between(18, 100)]
    lg.add("Idade impossível", "Idade fora de 18–100 anos (ex.: 0, 999)", "Remover", before, len(df), before - len(df))
    return df


def _regulatorio(df, lg):
    df = df.copy()
    parcela = R.parcela_price(df["valor_contratado"], df["taxa_juros_am"], df["prazo_meses"])
    margem = df["modalidade"].map(R.MARGEM_CONSIGNAVEL)
    checks = [
        ("Acima do teto do INSS", f"Consignado INSS com juros acima de {num(R.TETO_JUROS_INSS_AM)}% a.m. (CNPS)",
         (df["modalidade"] == "Consignado INSS") & (df["taxa_juros_am"] > R.TETO_JUROS_INSS_AM)),
        ("Margem consignável estourada", "Parcela acima de 45% (INSS/servidor) ou 35% (CLT) da renda",
         margem.notna() & df["renda_mensal"].notna() & (parcela / df["renda_mensal"] > margem + 0.005)),
        ("Sem elegibilidade", "Consignado contratado por quem não tem o vínculo (ex.: CLT no consignado INSS)",
         pd.Series([o not in R.ELEGIBILIDADE.get(m, {o}) for m, o in zip(df["modalidade"], df["ocupacao"])], index=df.index)),
    ]
    for step, rule, bad in checks:
        before = len(df)
        df = df[~bad.reindex(df.index, fill_value=False)]
        lg.add(step, rule + " — contrato irregular, fora do escopo do modelo", "Remover", before, len(df), before - len(df))
    return df


def _imputar_renda(df, lg):
    missing = df["renda_mensal"].isna()
    df["renda_imputada"] = missing.astype(int)
    df["renda_mensal"] = df["renda_mensal"].fillna(df.groupby("ocupacao")["renda_mensal"].transform("median")).round(2)
    lg.add("Renda ausente", "Mediana da renda dentro da mesma ocupação", "Imputar", len(df), len(df), int(missing.sum()))
    return df


def _imputar_score(df, lg):
    missing = df["score_bureau"].isna()
    df["sem_score"] = missing.astype(int)
    df["score_bureau"] = df["score_bureau"].fillna(df.groupby("rating")["score_bureau"].transform("median")).round()
    lg.add("Score ausente", "Cliente sem histórico no bureau: mediana do score no mesmo rating, com flag 'sem_score'",
           "Imputar", len(df), len(df), int(missing.sum()))
    return df


def _ausencia_estrutural(df, lg):
    apos = df["ocupacao"].eq("Aposentado/Pensionista INSS") & df["tempo_emprego_meses"].isna()
    df.loc[apos, "tempo_emprego_meses"] = 0
    lg.add("Ausência estrutural", "Aposentado não tem emprego atual: tempo de emprego = 0 (não é dado faltante)",
           "Preencher", len(df), len(df), int(apos.sum()))
    return df


def _tipagem(df, lg):
    for col in ("idade", "prazo_meses", "inadimplente", "score_bureau", "tempo_emprego_meses"):
        df[col] = df[col].round().astype("int64")
    df["data_contratacao"] = pd.to_datetime(df["data_contratacao"]).dt.strftime("%Y-%m-%d")
    lg.add("Tipagem", "Inteiros para contagens; data no padrão ISO (aaaa-mm-dd) para o banco", "Converter",
           len(df), len(df), 0)
    return df


STEPS = (_padronizar, _duplicatas, _moeda, _datas, _idade, _imputar_renda, _regulatorio, _imputar_score,
         _ausencia_estrutural, _tipagem)


# ---------------------------------------------------------------- atributos
AGE_BINS, AGE_LABELS = [17, 25, 35, 45, 60, 200], ["18–25", "26–35", "36–45", "46–60", "60+"]
SM_BINS = [0, 1, 2, 3, 5, 10, np.inf]
SM_LABELS = ["até 1 SM", "1–2 SM", "2–3 SM", "3–5 SM", "5–10 SM", "10+ SM"]
REGIAO = {u: r for u, (_, r, _) in R.UFS.items()}

FEATURE_DOCS = {
    "parcela": "Parcela mensal pela Tabela Price",
    "comprometimento_renda": "Parcela ÷ renda mensal",
    "renda_em_sm": f"Renda em salários mínimos (R$ {num(R.SALARIO_MINIMO)})",
    "taxa_aa": "Taxa ao mês convertida para ao ano: (1 + i)¹² − 1",
    "regiao": "Região do país a partir da UF",
    "faixa_etaria": "Faixa etária",
    "faixa_renda_sm": "Faixa de renda em salários mínimos",
    "rating_num": "Rating AA–H convertido em 1–9",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["parcela"] = R.parcela_price(out["valor_contratado"], out["taxa_juros_am"], out["prazo_meses"]).round(2)
    out["comprometimento_renda"] = (out["parcela"] / out["renda_mensal"]).round(4)
    out["renda_em_sm"] = (out["renda_mensal"] / R.SALARIO_MINIMO).round(3)
    out["taxa_aa"] = (((1 + out["taxa_juros_am"] / 100) ** 12 - 1) * 100).round(2)
    out["regiao"] = out["uf"].map(REGIAO)
    out["faixa_etaria"] = pd.cut(out["idade"], AGE_BINS, labels=AGE_LABELS).astype(str)
    out["faixa_renda_sm"] = pd.cut(out["renda_em_sm"], SM_BINS, labels=SM_LABELS, right=False).astype(str)
    out["rating_num"] = out["rating"].map({g: i for i, g in enumerate(R.RATINGS, start=1)})
    return out


def revenue_and_loss(preds: pd.DataFrame, lgd: float):
    juros_12m = R.juros_primeiros_meses(preds["valor_contratado"], preds["taxa_juros_am"], preds["prazo_meses"], 12)
    return np.asarray(juros_12m), preds["valor_contratado"].to_numpy() * lgd


def prepare_payload(payload: dict) -> pd.DataFrame:
    return add_features(pd.DataFrame([payload]))


def insights(clean: pd.DataFrame, policy: pd.DataFrame) -> list[tuple[str, str]]:
    out = []
    g = clean.groupby("rating")["inadimplente"].mean().reindex(R.RATINGS).dropna()
    if len(g) > 1:
        out.append(("fa-ranking-star", f"O rating ordena o risco: inadimplência de <b>{pct(g.iloc[0])}</b> no {g.index[0]} "
                                       f"contra <b>{pct(g.iloc[-1])}</b> no {g.index[-1]}."))
    m = clean.groupby("modalidade")["inadimplente"].mean().sort_values()
    if len(m) > 1:
        out.append(("fa-building-columns", f"Modalidade importa: <b>{m.index[0]}</b> tem a menor inadimplência "
                                           f"(<b>{pct(m.iloc[0])}</b>) e <b>{m.index[-1]}</b>, a maior (<b>{pct(m.iloc[-1])}</b>). "
                                           "O desconto em folha reduz o risco."))
    if len(clean) > 400:
        q = clean.groupby("modalidade")["comprometimento_renda"].transform(
            lambda s: pd.qcut(s.rank(method="first"), 4, labels=False) if len(s) >= 8 else 0)
        top, low = clean.loc[q == 3, "inadimplente"].mean(), clean.loc[q == 0, "inadimplente"].mean()
        if pd.notna(top) and pd.notna(low):
            out.append(("fa-percent", "Paradoxo de Simpson no comprometimento de renda: no agregado ele quase não separa o risco, "
                                      "porque os consignados têm parcela alta e inadimplência baixa. Dentro de cada modalidade, "
                                      f"o quartil mais comprometido inadimple <b>{pct(top)}</b>, contra <b>{pct(low)}</b> no menos comprometido."))
    r = clean.groupby("possui_restricao")["inadimplente"].mean()
    if {"Sim", "Não"} <= set(r.index):
        out.append(("fa-user-slash", f"Cliente negativado inadimple <b>{pct(r['Sim'])}</b> das vezes; sem restrição, "
                                     f"<b>{pct(r['Não'])}</b>."))
    return out


def _example():
    return {"idade": 34, "ocupacao": "Assalariado CLT", "renda_mensal": 3200.0, "tempo_emprego_meses": 30,
            "score_bureau": 640, "possui_restricao": "Não", "uf": "SP", "modalidade": "Consignado privado (CLT)",
            "valor_contratado": 6000.0, "prazo_meses": 24, "taxa_juros_am": 3.2, "rating": "B"}


SPEC = DatasetSpec(
    id="br", name="Crédito PF Brasil (sintética, regras 2026)", short="Brasil", currency="R$",
    raw_file="credito_brasil_sintetico.csv", is_synthetic=True,
    source_url="https://www.bcb.gov.br/estatisticas/estatisticasmonetariascredito",
    source_label="Gerada no projeto, calibrada com estatísticas do Banco Central",
    license="Base gerada pelo projeto (uso livre)",
    source_html=("<b>Base sintética</b> de 30 mil propostas de crédito pessoa física, gerada com semente fixa por "
                 "<code>br_generator.py</code>. Não existe base pública brasileira contrato a contrato com o desfecho "
                 "de pagamento: o SCR do Banco Central é protegido por sigilo bancário. A geração segue as regras de "
                 "2026 (teto do consignado INSS, margem consignável, salário mínimo, prazos) e foi calibrada com taxas "
                 "e inadimplência por modalidade divulgadas pelo BC."),
    read_raw=read_raw,
    expected_columns=tuple(COLUMN_DOCS), column_docs=COLUMN_DOCS,
    target="inadimplente", grade="rating", grade_order=tuple(R.RATINGS),
    amount="valor_contratado", income="renda_mensal", income_label="Renda mensal (R$)",
    rate="taxa_juros_am", rate_label="Taxa de juros (% a.m.)",
    purpose="modalidade", purpose_label="Modalidade", age="idade",
    build_schema=build_schema, rules=RULES, validation_view=validation_view,
    cleaning_steps=STEPS, add_features=add_features, feature_docs=FEATURE_DOCS,
    numeric_features=("idade", "renda_mensal", "tempo_emprego_meses", "score_bureau", "valor_contratado", "prazo_meses",
                      "taxa_juros_am", "parcela", "comprometimento_renda", "renda_em_sm"),
    categorical_features=("ocupacao", "modalidade", "possui_restricao", "rating", "regiao"),
    imputed={"renda_imputada": ("renda_mensal", "comprometimento_renda", "renda_em_sm"), "sem_score": ("score_bureau",)},
    policy_cols=("valor_contratado", "taxa_juros_am", "prazo_meses", "rating", "modalidade"),
    revenue_and_loss=revenue_and_loss, default_cutoff=0.20,
    form_fields=(
        FormField("idade", "Idade", "int", 34, 18, 100),
        FormField("ocupacao", "Ocupação", "select", "Assalariado CLT", options=tuple(R.OCUPACOES)),
        FormField("renda_mensal", "Renda mensal (R$)", "float", 3200.0, 500.0, 100_000.0, 100.0),
        FormField("tempo_emprego_meses", "Meses no emprego", "int", 30, 0, 600),
        FormField("score_bureau", "Score de bureau", "int", 640, 0, 1000, 10),
        FormField("possui_restricao", "Nome com restrição", "select", "Não", options=("Não", "Sim")),
        FormField("uf", "UF", "select", "SP", options=tuple(sorted(R.UFS))),
        FormField("modalidade", "Modalidade", "select", "Consignado privado (CLT)", options=tuple(R.MODALIDADES)),
        FormField("valor_contratado", "Valor (R$)", "float", 6000.0, 300.0, 500_000.0, 500.0),
        FormField("prazo_meses", "Prazo (meses)", "int", 24, 1, 108),
        FormField("taxa_juros_am", "Juros (% a.m.)", "float", 3.2, 0.5, 15.0, 0.05),
        FormField("rating", "Rating", "select", "B", options=tuple(R.RATINGS)),
    ),
    prepare_payload=prepare_payload, example_payload=_example(), predict_route="/br/predict",
    segment_dims={"rating": "Rating", "modalidade": "Modalidade", "ocupacao": "Ocupação", "regiao": "Região",
                  "faixa_renda_sm": "Renda em salários mínimos", "faixa_etaria": "Faixa etária",
                  "possui_restricao": "Nome com restrição"},
    dim_orders={"rating": R.RATINGS, "faixa_etaria": AGE_LABELS, "faixa_renda_sm": SM_LABELS, "regiao": R.REGIOES},
    dist_vars={"comprometimento_renda": "Comprometimento de renda", "score_bureau": "Score de bureau",
               "taxa_juros_am": "Taxa de juros (% a.m.)", "renda_mensal": "Renda mensal (R$)",
               "valor_contratado": "Valor contratado (R$)", "idade": "Idade"},
    corr_cols={"inadimplente": "default", "comprometimento_renda": "% renda", "score_bureau": "score", "rating_num": "rating",
               "taxa_juros_am": "taxa", "renda_mensal": "renda", "valor_contratado": "valor", "prazo_meses": "prazo",
               "idade": "idade"},
    impossible_vars=(("idade", "Idade (anos)"), ("taxa_juros_am", "Juros do consignado INSS (% a.m.)")),
    compare_vars={"renda_mensal": "Renda mensal (R$)", "idade": "Idade (anos)", "score_bureau": "Score de bureau",
                  "taxa_juros_am": "Taxa de juros (% a.m.)"},
    impute_view=("score_bureau", "sem_score", "Imputação do score por rating", "Score (0–1000)"),
    scatter=dict(x="renda_mensal", y="parcela", xlabel="Renda mensal (R$)", ylabel="Parcela (R$)",
                 ratios=(0.35, 0.45), ratio_label="margem {p}%", ymax=None,
                 note="As linhas marcam as margens consignáveis (35% CLT, 45% INSS/servidor). Os consignados ficam abaixo "
                      "delas; nas demais modalidades não há limite legal de parcela. A faixa vertical à esquerda é o piso "
                      "do salário mínimo para rendas formais."),
    insights=insights,
    texts={
        "missing_note": "Tempo de emprego vazio é esperado para aposentados (ausência estrutural, não erro): por isso a coluna "
                        "não entra na regra de completude.",
        "ingestion_note": "Repare nos formatos brasileiros: renda em texto (<code>R$ 3.450,00</code>), data dd/mm/aaaa, "
                          "CSV separado por ponto e vírgula. Algumas linhas trazem UF minúscula, datas ISO e "
                          "<code>S</code>/<code>N</code> no lugar de Sim/Não.",
        "quality_note": "Idades como 0 e 999 são erro de digitação. No consignado INSS, taxas acima de "
                        f"{num(R.TETO_JUROS_INSS_AM)}% a.m. violam o teto do CNPS: o contrato é irregular e sai da base.",
        "iv_note": "<b>rating</b> e <b>score_bureau</b> concentram a informação porque resumem o histórico do cliente. "
                   "A taxa de juros também é forte: o banco cobra mais de quem considera mais arriscado.",
        "cleaning_intro": "Regras aplicadas em ordem, incluindo as regulatórias brasileiras, todas registradas no <code>cleaning_log</code>.",
        "policy_intro": "Resultado de 12 meses: juros das 12 primeiras parcelas (Tabela Price) recebidos dos bons pagadores "
                        "menos a perda (LGD × valor) dos inadimplentes aprovados.",
    },
)
