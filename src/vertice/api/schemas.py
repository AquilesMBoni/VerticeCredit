"""Contratos de entrada e saída da API (Pydantic v2)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Applicant(BaseModel):
    person_age: int = Field(..., ge=18, le=100, description="Idade em anos", examples=[29])
    person_income: float = Field(..., gt=0, description="Renda anual em USD", examples=[52000])
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"] = Field(..., examples=["RENT"])
    person_emp_length: float = Field(..., ge=0, le=60, description="Anos de emprego", examples=[4])
    loan_intent: Literal["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION",
                         "HOMEIMPROVEMENT"] = Field(..., examples=["EDUCATION"])
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"] = Field(..., examples=["B"])
    loan_amnt: float = Field(..., gt=0, le=100_000, description="Valor solicitado em USD", examples=[9000])
    loan_int_rate: float = Field(..., ge=0, le=40, description="Taxa de juros anual (%)", examples=[11.5])
    cb_person_default_on_file: Literal["Y", "N"] = Field(..., examples=["N"])
    cb_person_cred_hist_length: int = Field(..., ge=0, le=60, examples=[5])


class Prediction(BaseModel):
    probability_default: float
    risk_band: str
    decision: Literal["Aprovar", "Recusar"]
    cutoff: float
    model_id: str
    model: str
    dataset: str = "us"


class RunResult(BaseModel):
    run_id: str
    dataset: str
    status: str
    duration_s: float


class Health(BaseModel):
    status: str
    version: str
    datasets: dict[str, dict]


class PropostaBR(BaseModel):
    """Proposta de crédito pessoa física no Brasil. Regras regulatórias são checadas na entrada."""
    idade: int = Field(..., ge=18, le=100, examples=[34])
    ocupacao: Literal["Assalariado CLT", "Servidor público", "Aposentado/Pensionista INSS", "Autônomo",
                      "Empresário/MEI"] = Field(..., examples=["Assalariado CLT"])
    renda_mensal: float = Field(..., gt=0, description="Renda mensal bruta (R$)", examples=[3200.0])
    tempo_emprego_meses: int = Field(0, ge=0, le=900, examples=[30])
    score_bureau: int = Field(..., ge=0, le=1000, examples=[640])
    possui_restricao: Literal["Sim", "Não"] = Field(..., examples=["Não"])
    uf: str = Field(..., min_length=2, max_length=2, examples=["SP"])
    modalidade: Literal["Consignado INSS", "Consignado público", "Consignado privado (CLT)", "Pessoal não consignado",
                        "Veículos"] = Field(..., examples=["Consignado privado (CLT)"])
    valor_contratado: float = Field(..., gt=0, le=2_000_000, examples=[6000.0])
    prazo_meses: int = Field(..., ge=1, le=120, examples=[24])
    taxa_juros_am: float = Field(..., gt=0, le=20, description="Juros ao mês (%)", examples=[3.2])
    rating: Literal["AA", "A", "B", "C", "D", "E", "F", "G", "H"] = Field(..., examples=["B"])

    @model_validator(mode="after")
    def regras_brasileiras(self):
        from ..datasets import br_rules as R
        from ..datasets._fmt import num

        erros = []
        if self.uf.upper() not in R.UFS:
            erros.append(f"UF inválida: {self.uf}")
        self.uf = self.uf.upper()
        eleg = R.ELEGIBILIDADE.get(self.modalidade)
        if eleg and self.ocupacao not in eleg:
            erros.append(f"{self.modalidade} exige vínculo: {', '.join(sorted(eleg))}")
        if self.modalidade == "Consignado INSS" and self.taxa_juros_am > R.TETO_JUROS_INSS_AM:
            erros.append(f"Juros acima do teto do consignado INSS ({num(R.TETO_JUROS_INSS_AM)}% a.m.)")
        if self.prazo_meses > R.PRAZO_MAXIMO[self.modalidade]:
            erros.append(f"Prazo acima do máximo da modalidade ({R.PRAZO_MAXIMO[self.modalidade]} meses)")
        margem = R.MARGEM_CONSIGNAVEL.get(self.modalidade)
        if margem:
            parcela = float(R.parcela_price(self.valor_contratado, self.taxa_juros_am, self.prazo_meses))
            if parcela > margem * self.renda_mensal + 0.01:
                erros.append(f"Parcela de R$ {num(parcela)} excede a margem consignável de {margem:.0%} da renda")
        if erros:
            raise ValueError("; ".join(erros))
        return self
