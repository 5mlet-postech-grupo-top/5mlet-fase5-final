from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from .utils import DATA_DIR, logger


def list_xlsx(data_dir: Path | None = None) -> List[Path]:
    d = data_dir or DATA_DIR
    if not d.exists():
        return []
    return sorted([p for p in d.glob("*.xlsx") if p.is_file()])


def load_all_training_data(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and concatenate all .xlsx files inside ./data."""
    files = list_xlsx(data_dir)
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {data_dir or DATA_DIR}")

    dfs = []
    for f in files:
        try:
            df = pd.read_excel(f)
            df["__source_file__"] = f.name
            dfs.append(df)
            logger.info("data_loaded", extra={"file": f.name, "rows": len(df), "cols": len(df.columns)})
        except Exception as e:
            logger.exception("data_load_failed", extra={"file": str(f), "error": str(e)})

    if not dfs:
        raise RuntimeError("No datasets could be loaded.")
    return pd.concat(dfs, ignore_index=True, sort=False)
