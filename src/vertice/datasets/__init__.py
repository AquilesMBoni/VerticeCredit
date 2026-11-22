"""Registro das bases disponíveis."""
from .br import SPEC as BR
from .us import SPEC as US

DATASETS = {US.id: US, BR.id: BR}

__all__ = ["DATASETS", "US", "BR"]
