from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd


def metadata_path_for(target: Path) -> Path:
    return target.with_suffix(f"{target.suffix}.sync.json")


def load_sync_metadata(target: Path) -> dict[str, object] | None:
    metadata_path = metadata_path_for(target)
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def write_sync_metadata(
    target: Path,
    *,
    synced_end_date: str,
    extra: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "synced_end_date": pd.Timestamp(synced_end_date).date().isoformat(),
    }
    if extra:
        payload.update(extra)
    metadata_path_for(target).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def infer_csv_end_timestamp(target: Path, *, date_columns: Sequence[str]) -> pd.Timestamp | None:
    if not target.exists():
        return None

    latest: pd.Timestamp | None = None
    for column in date_columns:
        try:
            frame = pd.read_csv(target, usecols=[column])
        except ValueError:
            continue
        values = pd.to_datetime(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        candidate = pd.Timestamp(values.max())
        if latest is None or candidate > latest:
            latest = candidate
    return latest


def csv_target_is_fresh(
    target: Path,
    *,
    end_date: str,
    date_columns: Sequence[str],
    metadata_hint: str | None = None,
) -> bool:
    if not target.exists():
        return False

    metadata = load_sync_metadata(target)
    if metadata is not None:
        synced_end_date = metadata.get("synced_end_date")
        if synced_end_date and pd.Timestamp(str(synced_end_date)) >= pd.Timestamp(end_date):
            return True

    latest = infer_csv_end_timestamp(target, date_columns=date_columns)
    if latest is None or latest < pd.Timestamp(end_date):
        return False
    write_sync_metadata(
        target,
        synced_end_date=end_date,
        extra={"date_column": metadata_hint or str(date_columns[0])},
    )
    return True
