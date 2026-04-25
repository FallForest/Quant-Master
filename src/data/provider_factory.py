from __future__ import annotations

from data.providers.csv_provider import CsvDataProvider
from data.providers.duckdb_provider import DuckDBDataProvider
from data.providers.parquet_provider import ParquetDataProvider


def build_data_provider(
    *,
    provider_name: str,
    data_root: str,
    universe_root: str,
    adjust: str,
):
    common_kwargs = {
        "base_path": data_root,
        "universe_root": universe_root,
        "default_adjust": adjust,
    }
    if provider_name == "csv":
        return CsvDataProvider(**common_kwargs)
    if provider_name == "parquet":
        return ParquetDataProvider(**common_kwargs)
    if provider_name == "duckdb":
        return DuckDBDataProvider(**common_kwargs)
    raise ValueError(f"Unsupported data provider: {provider_name}")
