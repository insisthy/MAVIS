from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Any

import pandas as pd

CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "gb18030")


def read_csv_auto(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a CSV with common UTF and Chinese Windows encodings."""
    p = Path(path)
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(p, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(p, **kwargs)


def read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    """Read a CSV or return an empty DataFrame when it does not exist."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return read_csv_auto(p)


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return p


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
