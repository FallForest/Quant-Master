from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.providers.local_file_support import LocalFileProviderSupport


class ParquetDataProvider(LocalFileProviderSupport):
    file_suffix = ".parquet"

    def read_data(self, file_path: Path) -> pd.DataFrame:
        try:
            return pd.read_parquet(file_path)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet support requires `pyarrow`. Run `py -m pip install -r requirements.txt` first."
            ) from exc
