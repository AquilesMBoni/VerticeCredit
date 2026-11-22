"""Gerador da base brasileira de crédito pessoa física (sintética e reprodutível).

Por que sintética? Não existe base pública brasileira com contrato a contrato e o
desfecho de pagamento: o SCR do Banco Central é protegido por sigilo bancário
(LC 105/2001) e o SCR.data só publica agregados. Por isso a base é gerada com
semente fixa, seguindo as regras e calibrada por estatísticas públicas de 2026:

* taxas por modalidade próximas às médias do BC, respeitando o teto do INSS;
* inadimplência-alvo por modalidade próxima das taxas divulgadas pelo BC;
* margem consignável, elegibilidade, prazo máximo e piso do salário mínimo;
* distribuição de UFs proporcional à população (Censo 2022).

A sujeira é injetada de propósito, com os problemas típicos de dados brasileiros:
moeda em texto ("R$ 3.450,00"), datas em dois formatos, UF minúscula ou por
extenso, "Sim/Não/S/N", duplicatas e contratos que violam a regulação.

    python -m vertice.datasets.br_generator
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import br_rules as R

SEED = 2026
N = 30_000

OCUP_SHARE = {"Assalariado CLT": .38, "Servidor público": .10, "Aposentado/Pensionista INSS": .20,
              "Autônomo": .20, "Empresário/MEI": .12}
MOD_BY_OCUP = {
    "Aposentado/Pensionista INSS": {"Consignado INSS": .60, "Pessoal não consignado": .25, "Veículos": .15},
    "Servidor público": {"Consignado público": .60, "Pessoal não consignado": .20, "Veículos": .20},
    "Assalariado CLT": {"Consignado privado (CLT)": .40, "Pessoal não consignado": .35, "Veículos": .25},
    "Autônomo": {"Pessoal não consignado": .70, "Veículos": .30},
    "Empresário/MEI": {"Pessoal não consignado": .60, "Veículos": .40},
}
RENDA_MEDIANA = {"Assalariado CLT": 2900, "Servidor público": 6200, "Aposentado/Pensionista INSS": 2300,
                 "Autônomo": 2600, "Empresário/MEI": 4800}
# taxa base (% a.m.) e prazo mínimo por modalidade
TAXA_BASE = {"Consignado INSS": 1.72, "Consignado público": 1.85, "Consignado privado (CLT)": 3.4,
             "Pessoal não consignado": 6.0, "Veículos": 2.2}
PRAZO_MIN = {"Consignado INSS": 12, "Consignado público": 12, "Consignado privado (CLT)": 6,
             "Pessoal não consignado": 3, "Veículos": 24}
# inadimplência-alvo por modalidade (aproximação das taxas publicadas pelo BC em 2026)
ALVO_DEFAULT = {"Consignado INSS": .040, "Consignado público": .030, "Consignado privado (CLT)": .105,
                "Pessoal não consignado": .13, "Veículos": .068}
RATING_SHARE = [.08, .15, .20, .20, .14, .10, .06, .04, .03]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _brl(v: float) -> str:
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def generate(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ocup = rng.choice(list(OCUP_SHARE), size=n, p=list(OCUP_SHARE.values()))
    modal = np.array([rng.choice(list(MOD_BY_OCUP[o]), p=list(MOD_BY_OCUP[o].values())) for o in ocup])

    ufs = list(R.UFS)
    w = np.array([R.UFS[u][2] for u in ufs])
    uf = rng.choice(ufs, size=n, p=w / w.sum())

    aposentado = ocup == "Aposentado/Pensionista INSS"
    idade = np.where(aposentado, rng.normal(67, 8, n).clip(50, 92), rng.gamma(6, 6.2, n).clip(0, 52) + 18).round().astype(int)

    q = rng.normal(0, 1, n)  # qualidade de crédito latente (não observada)
    med = np.array([RENDA_MEDIANA[o] for o in ocup])
    renda = med * np.exp(rng.normal(0.25 * q, 0.55, n))
    formal = np.isin(ocup, list(R.RENDA_FORMAL))
    renda = np.where(formal, np.maximum(renda, R.SALARIO_MINIMO), np.maximum(renda, 500)).round(2)

    max_emp = np.maximum((idade - 16) * 12, 0)
    tempo = np.where(aposentado, np.nan, np.minimum(rng.gamma(1.6, 40 + 8 * (idade - 18) / 4, n), max_emp)).round()

    score = (620 + 140 * q + rng.normal(0, 70, n)).clip(0, 1000).round()
    thin = rng.random(n) < np.where(idade < 25, 0.16, 0.04)
    score = np.where(thin, np.nan, score)
    restr = rng.random(n) < _sigmoid(-2.0 - 1.4 * q)

    # rating interno AA–H (modelo do banco: vê o score e a restrição, com ruído)
    risk_view = -q + 0.9 * restr + rng.normal(0, 0.55, n)
    cuts = np.quantile(risk_view, np.cumsum(RATING_SHARE)[:-1])
    rating = np.array(R.RATINGS)[np.searchsorted(cuts, risk_view)]
    rnum = np.searchsorted(cuts, risk_view)

    taxa = np.array([TAXA_BASE[m] for m in modal]) * (1 + 0.07 * (rnum - 3)) * np.exp(rng.normal(0, 0.06, n))
    taxa = np.where(modal == "Consignado INSS", np.minimum(taxa, R.TETO_JUROS_INSS_AM), taxa)
    taxa = np.where(modal == "Consignado público", np.minimum(taxa, 2.05), taxa).round(2)

    pmax = np.array([R.PRAZO_MAXIMO[m] for m in modal])
    pmin = np.array([PRAZO_MIN[m] for m in modal])
    prazo = (pmin + rng.beta(2.2, 1.6, n) * (pmax - pmin)).round().astype(int)
    prazo = np.where(modal == "Veículos", np.choose(rng.integers(0, 4, n), [24, 36, 48, 60]), prazo)

    margem = np.array([R.MARGEM_CONSIGNAVEL.get(m, np.nan) for m in modal])
    util = np.where(np.isnan(margem), np.nan, rng.beta(3, 2, n) * margem * 0.98)
    frac_livre = np.where(modal == "Veículos", rng.beta(2.5, 6, n) * 0.55, rng.beta(1.8, 7, n) * 0.6)
    frac = np.where(np.isnan(util), frac_livre, util)
    parcela = renda * frac
    i = taxa / 100
    valor = (parcela * (1 - (1 + i) ** (-prazo)) / i).round(-1).clip(300, None)
    parcela = R.parcela_price(valor, taxa, prazo)
    compr = parcela / renda

    jovem = (idade < 25).astype(float)
    informal = np.isin(ocup, ["Autônomo"]).astype(float)
    base = (-1.05 * q + 1.0 * restr + 5.0 * (compr - 0.25) + 0.45 * informal + 0.35 * jovem
            - 0.25 * np.log(renda / R.SALARIO_MINIMO) + 0.08 * (taxa - 2) + rng.normal(0, 0.35, n))

    y = np.zeros(n, dtype=int)
    for m, alvo in ALVO_DEFAULT.items():
        idx = modal == m
        lo, hi = -12.0, 6.0
        for _ in range(60):  # bisseção: intercepto que atinge a inadimplência-alvo
            mid = (lo + hi) / 2
            if _sigmoid(mid + base[idx]).mean() > alvo:
                hi = mid
            else:
                lo = mid
        y[idx] = (rng.random(idx.sum()) < _sigmoid(mid + base[idx])).astype(int)

    start = date(2025, 1, 1)
    dias = rng.integers(0, 546, n)  # contratos de jan/2025 a jun/2026 (12 meses de observação)
    datas = [start + timedelta(days=int(d)) for d in dias]

    df = pd.DataFrame({
        "id_proposta": np.arange(100001, 100001 + n),
        "data_contratacao": [d.strftime("%d/%m/%Y") for d in datas],
        "uf": uf, "idade": idade, "ocupacao": ocup,
        "renda_mensal": [_brl(v) for v in renda],
        "tempo_emprego_meses": tempo,
        "score_bureau": score,
        "possui_restricao": np.where(restr, "Sim", "Não"),
        "modalidade": modal,
        "valor_contratado": valor,
        "prazo_meses": prazo,
        "taxa_juros_am": taxa,
        "rating": rating,
        "inadimplente": y,
    })
    return _inject_dirt(df, rng)


def _inject_dirt(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    df = df.copy()

    def pick(frac: float, mask=None) -> np.ndarray:
        pool = np.flatnonzero(mask) if mask is not None else np.arange(n)
        k = max(1, int(round(frac * n)))
        return rng.choice(pool, size=min(k, len(pool)), replace=False)

    ix = pick(.025)
    df.loc[ix, "uf"] = [u.lower() if j % 2 else f" {u} " for j, u in enumerate(df.loc[ix, "uf"])]
    ix = pick(.004)
    df.loc[ix, "uf"] = [R.UFS[u.strip().upper()][0] for u in df.loc[ix, "uf"]]

    ix = pick(.02)
    df.loc[ix, "ocupacao"] = [o.lower() if j % 2 else o.upper() + " " for j, o in enumerate(df.loc[ix, "ocupacao"])]

    ix = pick(.30)
    df.loc[ix, "possui_restricao"] = df.loc[ix, "possui_restricao"].map({"Sim": "S", "Não": "N"})
    ix = pick(.02)
    df.loc[ix, "possui_restricao"] = df.loc[ix, "possui_restricao"].str.lower()

    ix = pick(.01)
    df.loc[ix, "rating"] = df.loc[ix, "rating"].str.lower()

    ix = pick(.025)
    df.loc[ix, "renda_mensal"] = ""
    ix = pick(.01)
    df.loc[ix, "renda_mensal"] = [s.replace("R$ ", "").replace(".", "").replace(",", ".") for s in df.loc[ix, "renda_mensal"]]
    ix = pick(.003)
    df.loc[ix, "renda_mensal"] = "R$ 0,00"

    ix = pick(.02)
    df.loc[ix, "data_contratacao"] = [pd.to_datetime(d, dayfirst=True).strftime("%Y-%m-%d") for d in df.loc[ix, "data_contratacao"]]
    ix = pick(.004)
    df.loc[ix, "data_contratacao"] = rng.choice(["31/02/2026", "30/02/2025", "00/05/2025", "15/13/2025"], size=len(ix))
    ix = pick(.002)
    df.loc[ix, "data_contratacao"] = rng.choice(["10/03/2027", "22/08/2027"], size=len(ix))

    ix = pick(.001)
    df.loc[ix, "idade"] = rng.choice([0, 7, 16, 999, 135], size=len(ix))

    inss = (df["modalidade"] == "Consignado INSS").to_numpy()
    ix = pick(.004, inss)
    df.loc[ix, "taxa_juros_am"] = rng.choice([2.10, 2.35, 2.49], size=len(ix))

    consig = df["modalidade"].isin(R.MARGEM_CONSIGNAVEL).to_numpy()
    ix = pick(.004, consig)
    df.loc[ix, "valor_contratado"] = (df.loc[ix, "valor_contratado"] * 1.9).round(-1)

    clt = (df["ocupacao"] == "Assalariado CLT").to_numpy()
    ix = pick(.003, clt)
    df.loc[ix, "modalidade"] = "Consignado INSS"

    dup = df.sample(n=150, random_state=7)
    df = pd.concat([df, dup], ignore_index=True).sample(frac=1, random_state=11).reset_index(drop=True)
    return df


def main(path: Path | None = None) -> Path:
    from ..config import settings

    path = path or settings.raw_dir / "credito_brasil_sintetico.csv"
    df = generate()
    df.to_csv(path, index=False, sep=";", encoding="utf-8-sig")  # padrão de CSV brasileiro (Excel pt-BR)
    print(f"{len(df)} linhas → {path}")
    return path


if __name__ == "__main__":
    main()
