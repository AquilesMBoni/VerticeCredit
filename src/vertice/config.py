"""Configuração global do projeto (parâmetros que não dependem da base).

O que muda de uma base para outra (colunas, regras, textos) fica em
`vertice/datasets/<id>.py`. Caminhos podem ser sobrescritos por variáveis de
ambiente (12-factor), o que os testes usam para isolar o banco.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def _path(env: str, default: Path) -> Path:
    value = os.getenv(env)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Settings:
    raw_dir: Path = _path("VERTICE_RAW_DIR", ROOT_DIR / "data" / "raw")
    data_dir: Path = _path("VERTICE_DATA_DIR", ROOT_DIR / "data")
    models_root: Path = _path("VERTICE_MODELS", ROOT_DIR / "models")
    reports_root: Path = _path("VERTICE_REPORTS", ROOT_DIR / "reports")
    api_url: str = os.getenv("VERTICE_API_URL", "http://127.0.0.1:8000")
    default_dataset: str = os.getenv("VERTICE_DATASET", "us")

    random_state: int = 42
    test_size: float = 0.20
    cv_folds: int = 5

    def db_path(self, dataset_id: str) -> Path:
        return self.data_dir / f"vertice_{dataset_id}.db"

    def models_dir(self, dataset_id: str) -> Path:
        return _mkdir(self.models_root / dataset_id)

    def reports_dir(self, dataset_id: str) -> Path:
        return _mkdir(self.reports_root / dataset_id)

    def processed_dir(self, dataset_id: str) -> Path:
        return _mkdir(self.data_dir / "processed" / dataset_id)


def _mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


settings = Settings()
for _d in (settings.data_dir, settings.models_root, settings.reports_root):
    _d.mkdir(parents=True, exist_ok=True)
