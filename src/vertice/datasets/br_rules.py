"""Parâmetros e regras do crédito pessoa física no Brasil (referência: 2026).

Todos os números ficam aqui, com a fonte ao lado, para que a validação, a
limpeza, o gerador da base e o dashboard usem exatamente os mesmos valores.
Quando uma regra mudar (ex.: novo teto do CNPS), altere só este arquivo.
"""
from __future__ import annotations

# Decreto nº 12.797/2025: salário mínimo de R$ 1.621,00 a partir de 1º/1/2026
SALARIO_MINIMO = 1621.00

# Teto de juros do consignado INSS definido pelo CNPS: 1,85% ao mês (2025–2026)
TETO_JUROS_INSS_AM = 1.85

# Margem consignável (parcela máxima ÷ renda)
#   INSS e servidor público: 45% (a MP 1.355/2026 chegou a reduzir para 40%; perdeu validade)
#   Crédito do Trabalhador (CLT, consignado privado): 35% do salário
MARGEM_CONSIGNAVEL = {
    "Consignado INSS": 0.45,
    "Consignado público": 0.45,
    "Consignado privado (CLT)": 0.35,
}

# Prazo máximo (meses). INSS: 108 parcelas em 2026; demais são limites usuais de mercado.
PRAZO_MAXIMO = {
    "Consignado INSS": 108,
    "Consignado público": 96,
    "Consignado privado (CLT)": 48,
    "Pessoal não consignado": 36,
    "Veículos": 60,
}

# Quem pode contratar cada modalidade consignada
ELEGIBILIDADE = {
    "Consignado INSS": {"Aposentado/Pensionista INSS"},
    "Consignado público": {"Servidor público"},
    "Consignado privado (CLT)": {"Assalariado CLT"},
}

# Ocupações com renda formal (piso = salário mínimo)
RENDA_FORMAL = {"Assalariado CLT", "Servidor público", "Aposentado/Pensionista INSS"}

OCUPACOES = ["Assalariado CLT", "Servidor público", "Aposentado/Pensionista INSS", "Autônomo", "Empresário/MEI"]
MODALIDADES = list(PRAZO_MAXIMO)

# Níveis de risco AA–H: escala da Res. CMN 2.682/1999, revogada em 1º/1/2025 pela Res. CMN 4.966/2021,
# mas ainda usada no SCR (doc. 3040) como classificação de risco do cliente/operação.
RATINGS = ["AA", "A", "B", "C", "D", "E", "F", "G", "H"]

# Inadimplência: atraso superior a 90 dias (ativo problemático / estágio 3 na Res. CMN 4.966/2021)
DIAS_INADIMPLENCIA = 90

UFS = {
    "SP": ("São Paulo", "Sudeste", 44.4), "MG": ("Minas Gerais", "Sudeste", 20.5),
    "RJ": ("Rio de Janeiro", "Sudeste", 16.1), "ES": ("Espírito Santo", "Sudeste", 3.8),
    "BA": ("Bahia", "Nordeste", 14.1), "PE": ("Pernambuco", "Nordeste", 9.1),
    "CE": ("Ceará", "Nordeste", 8.8), "MA": ("Maranhão", "Nordeste", 6.8),
    "PB": ("Paraíba", "Nordeste", 4.0), "RN": ("Rio Grande do Norte", "Nordeste", 3.3),
    "AL": ("Alagoas", "Nordeste", 3.1), "PI": ("Piauí", "Nordeste", 3.3), "SE": ("Sergipe", "Nordeste", 2.2),
    "PR": ("Paraná", "Sul", 11.4), "RS": ("Rio Grande do Sul", "Sul", 10.9), "SC": ("Santa Catarina", "Sul", 7.6),
    "GO": ("Goiás", "Centro-Oeste", 7.1), "MT": ("Mato Grosso", "Centro-Oeste", 3.7),
    "MS": ("Mato Grosso do Sul", "Centro-Oeste", 2.8), "DF": ("Distrito Federal", "Centro-Oeste", 2.8),
    "PA": ("Pará", "Norte", 8.1), "AM": ("Amazonas", "Norte", 3.9), "RO": ("Rondônia", "Norte", 1.6),
    "TO": ("Tocantins", "Norte", 1.5), "AC": ("Acre", "Norte", 0.83), "AP": ("Amapá", "Norte", 0.73),
    "RR": ("Roraima", "Norte", 0.64),
}
REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]


def parcela_price(valor, taxa_am_pct, prazo):
    """Parcela pela Tabela Price (sistema francês), padrão no crédito pessoal brasileiro."""
    import numpy as np

    i = np.asarray(taxa_am_pct, dtype=float) / 100
    n = np.asarray(prazo, dtype=float)
    v = np.asarray(valor, dtype=float)
    return v * i / (1 - (1 + i) ** (-n))


def juros_primeiros_meses(valor, taxa_am_pct, prazo, meses: int = 12):
    """Juros efetivamente pagos nas primeiras `meses` parcelas da Tabela Price."""
    import numpy as np

    i = np.asarray(taxa_am_pct, dtype=float) / 100
    n = np.asarray(prazo, dtype=float)
    v = np.asarray(valor, dtype=float)
    k = np.minimum(meses, n)
    pmt = parcela_price(v, taxa_am_pct, n)
    saldo_k = v * (1 + i) ** k - pmt * ((1 + i) ** k - 1) / i
    return k * pmt - (v - saldo_k)
