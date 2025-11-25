# metrics/loggig_excel.py

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import pandas as pd


def log_to_excel(
    rows: List[Dict],
    excel_path: str = "logs/rag_eval_log.xlsx",
) -> None:
    """
    Append evaluation rows to an Excel file.
    Each row is a dict with simple Python types (str/int/float/bool).

    - If the file doesn't exist, it will be created with a header.
    - If it exists, new rows are appended.
    """
    path = Path(excel_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(rows)

    if path.exists():
        # Append to existing file
        existing_df = pd.read_excel(path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined.to_excel(path, index=False)
    else:
        # Create new file
        new_df.to_excel(path, index=False)
