"""Testes usam diretórios temporários: nunca tocam nos bancos do projeto."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_tmp = Path(tempfile.mkdtemp(prefix="vertice-tests-"))
os.environ["VERTICE_DATA_DIR"] = str(_tmp / "data")
os.environ["VERTICE_MODELS"] = str(_tmp / "models")
os.environ["VERTICE_REPORTS"] = str(_tmp / "reports")
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def raw_us():
    from vertice import context
    from vertice.pipeline import ingestion
    with context.use("us"):
        return ingestion.ingest().df


@pytest.fixture(scope="session")
def raw_br():
    from vertice import context
    from vertice.pipeline import ingestion
    with context.use("br"):
        return ingestion.ingest().df


@pytest.fixture(scope="session")
def pipeline_run():
    from vertice.pipeline import runner
    return {ds: runner.run(dataset=ds) for ds in ("us", "br")}
