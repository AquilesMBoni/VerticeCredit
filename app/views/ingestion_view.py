from __future__ import annotations

import pandas as pd
import streamlit as st

import data
import ui
from vertice import context
from vertice.pipeline.ingestion import raw_path


def render(mode: str) -> None:
    spec = context.spec()
    ui.page_header("fa-file-import", "Fase 1 de 8", "Ingestão",
                   "O arquivo bruto é lido sem nenhuma alteração de conteúdo, recebe uma impressão digital SHA-256 "
                   "e é gravado na tabela <code>raw_loans</code> do SQLite. A camada bruta é imutável: toda correção "
                   "acontece nas fases seguintes, com registro.")
    grades = ui.grade_filter(spec)

    run = data.overview()["run"]
    raw = data.tg("raw_loans", grades).drop(columns=["record_id"])
    path = raw_path()
    size_mb = path.stat().st_size / 1e6 if path.exists() else 0
    num_cols = sum(pd.api.types.is_numeric_dtype(raw[c]) for c in raw.columns)

    ui.kpi_row([
        ui.kpi("fa-table-list", "Linhas", ui.fmt_int(len(raw)), "registros" + ui.filter_note(grades)),
        ui.kpi("fa-table-columns", "Colunas", str(raw.shape[1]), f"{num_cols} numéricas, {raw.shape[1] - num_cols} em texto"),
        ui.kpi("fa-weight-hanging", "Tamanho do arquivo", f"{ui.fmt_num(size_mb, 2)} MB", path.name),
        ui.kpi("fa-fingerprint", "SHA-256", f'<span style="font-size:1.05rem">{run["dataset_sha256"][:12]}…</span>',
               "garante que a fonte não mudou entre execuções"),
    ])
    st.write("")

    c1, c2 = st.columns([1, 1.6])
    with c1:
        with ui.glass("source"):
            ui.section("fa-box-open", "Fonte dos dados")
            synth = ui.pill("warn", "dados sintéticos", "fa-flask") if spec.is_synthetic else ui.pill("good", "dados reais")
            st.markdown(f"""<div class="vx-note">{synth}<br><br>{spec.source_html}<br><br>
<i class="fa-solid fa-link"></i> <a href="{spec.source_url}" target="_blank">{ui.esc(spec.source_label)}</a><br>
<i class="fa-solid fa-scale-balanced"></i> Licença: {ui.esc(spec.license)}<br>
<i class="fa-solid fa-bullseye"></i> Alvo: <code>{spec.target}</code> (1 = inadimplente)<br>
<i class="fa-solid fa-folder-tree"></i> <code>data/raw/{spec.raw_file}</code></div>""", unsafe_allow_html=True)
    with c2:
        with ui.glass("dictionary"):
            ui.section("fa-book", "Dicionário de dados", "perfil da camada bruta")
            prof = pd.DataFrame({
                "coluna": raw.columns,
                "tipo": ["número" if pd.api.types.is_numeric_dtype(raw[c]) else "texto" for c in raw.columns],
                "descrição": [spec.column_docs.get(c, "") for c in raw.columns],
                "nulos": [int(raw[c].isna().sum() + (raw[c].astype(str).str.strip() == "").sum()) for c in raw.columns],
                "distintos": [int(raw[c].nunique()) for c in raw.columns],
            })
            ui.html_table(prof, formats={"nulos": ui.fmt_int, "distintos": ui.fmt_int}, max_height=520)

    st.write("")
    with ui.glass("sample"):
        ui.section("fa-eye", "Amostra da camada bruta", "primeiras 12 linhas, exatamente como chegaram")
        ui.html_table(raw.head(12))
        ui.note(spec.texts.get("ingestion_note", ""))

    st.write("")
    with ui.glass("describe"):
        ui.section("fa-calculator", "Estatística descritiva (colunas numéricas do bruto)")
        desc = raw.describe().T.reset_index().rename(columns={"index": "coluna", "count": "n", "mean": "média",
                                                              "std": "desvio", "min": "mín", "max": "máx"})
        fmt = {c: (lambda v: ui.fmt_num(v, 2)) for c in desc.columns if c != "coluna"}
        fmt["n"] = ui.fmt_int
        ui.html_table(desc, formats=fmt)
