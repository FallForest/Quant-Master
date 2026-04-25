from __future__ import annotations


def normalize_symbol(symbol: object) -> str:
    text = str(symbol).strip()
    return text.zfill(6) if text.isdigit() else text
