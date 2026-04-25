from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from data.providers.local_file_support import LocalFileProviderSupport


class DuckDBDataProvider(LocalFileProviderSupport):
    file_suffix = ".parquet"

    def __init__(
        self,
        base_path: str | Path = "data/lake",
        universe_root: str | Path = "data/universe",
        default_adjust: str = "qfq",
    ) -> None:
        super().__init__(base_path=base_path, universe_root=universe_root, default_adjust=default_adjust)
        try:
            import duckdb
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "DuckDB support requires `duckdb`. Run `py -m pip install -r requirements.txt` first."
            ) from exc
        self._duckdb = duckdb

    def read_data(self, file_path: Path) -> pd.DataFrame:
        connection = self._duckdb.connect(database=":memory:")
        try:
            return connection.execute(
                """
                SELECT *
                FROM read_parquet(?)
                ORDER BY timestamp
                """,
                [str(file_path)],
            ).df()
        finally:
            connection.close()
