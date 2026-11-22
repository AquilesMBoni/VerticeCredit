"""Formatação pt-BR usada nos textos gerados pelas bases."""


def pct(v: float, digits: int = 1) -> str:
    return f"{v * 100:,.{digits}f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def num(v: float, digits: int = 2) -> str:
    return f"{v:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")
