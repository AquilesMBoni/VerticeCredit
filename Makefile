# Atalhos (Linux/macOS). No Windows, rode os comandos equivalentes do README.
PY ?= python

install:
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install --no-deps -e .

pipeline:
	$(PY) -m vertice.pipeline.runner

api:
	uvicorn vertice.api.main:app --app-dir src --reload --port 8000

app:
	streamlit run app/app.py

test:
	$(PY) -m pytest

lint:
	ruff check src app tests

clean:
	rm -f data/vertice.db* models/*.joblib reports/*.json data/processed/*.csv data/processed/*.parquet

.PHONY: install pipeline api app test lint clean
