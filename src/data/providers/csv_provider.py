from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.providers.local_file_support import LocalFileProviderSupport


class CsvDataProvider(LocalFileProviderSupport):
    file_suffix = ".csv"

    def read_data(self, file_path: Path) -> pd.DataFrame:
        return pd.read_csv(file_path)
