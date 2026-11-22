"""Base ativa (EUA ou Brasil) para a requisição/execução atual.

Usa `ContextVar`, então cada thread (execução do Streamlit, requisição da API)
enxerga a própria base sem interferir nas outras.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from .config import settings

_current: ContextVar[str] = ContextVar("vertice_dataset", default=settings.default_dataset)


def current_id() -> str:
    return _current.get()


def set_dataset(dataset_id: str) -> None:
    from .datasets import DATASETS

    if dataset_id not in DATASETS:
        raise ValueError(f"Base desconhecida: {dataset_id}. Use uma de {list(DATASETS)}")
    _current.set(dataset_id)


@contextmanager
def use(dataset_id: str) -> Iterator[None]:
    token = _current.set(_current.get())
    try:
        set_dataset(dataset_id)
        yield
    finally:
        _current.reset(token)


def spec():
    from .datasets import DATASETS

    return DATASETS[current_id()]
