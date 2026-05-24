"""EDA (Exploratory Data Analysis) for MLE-Bench competitions.

Simplified port of the full _run_eda from the purple agent.
Produces structured metadata used by the tabular script generator.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

ENCODINGS = ["utf-8", "latin1", "cp1252"]
SAMPLE_ROWS = 100_000
HIGH_CARD_THRESHOLD = 50
SKEW_THRESHOLD = 5.0
DELIMITERS = ["/", "-", "_", "|", ":", ";"]


def run_eda(data_dir: Path) -> tuple[str, dict]:
    """Profile CSVs in data_dir and return (report_text, metadata_dict)."""
    import numpy as np
    import pandas as pd

    meta: dict = {
        "train_file": None,
        "test_file": None,
        "target_col": None,
        "target_cols": [],
        "id_col": None,
        "columns": {},
        "n_train_rows": 0,
        "target_nunique": 0,
        "target_dtype": "",
        "target_is_bool": False,
        "target_is_proba": False,
        "spend_cols": [],
        "binary_interaction_pairs": [],
        "numeric_interaction_pairs": [],
    }

    report_lines: list[str] = []

    # Find train/test CSVs
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists():
        # Try alternative names
        for candidate in sorted(data_dir.glob("*.csv")):
            if "train" in candidate.name.lower():
                train_path = candidate
                break
    if not test_path.exists():
        for candidate in sorted(data_dir.glob("*.csv")):
            if "test" in candidate.name.lower() and "sample" not in candidate.name.lower():
                test_path = candidate
                break

    if not train_path.exists():
        report_lines.append("ERROR: No train.csv found")
        return "\n".join(report_lines), meta

    meta["train_file"] = train_path.name
    meta["test_file"] = test_path.name if test_path.exists() else None

    # Load sample
    train_df = _read_resilient(train_path)
    if train_df is None:
        report_lines.append("ERROR: Could not read train.csv")
        return "\n".join(report_lines), meta

    test_df = _read_resilient(test_path) if test_path.exists() else None
    meta["n_train_rows"] = _count_rows(train_path)

    report_lines.append(f"TRAIN: {train_path.name} ({meta['n_train_rows']} rows, {len(train_df.columns)} cols)")
    if test_df is not None:
        report_lines.append(f"TEST: {test_path.name} ({_count_rows(test_path)} rows, {len(test_df.columns)} cols)")

    # Detect target column(s): columns in train but not in test
    if test_df is not None:
        train_only_cols = [c for c in train_df.columns if c not in test_df.columns]
    else:
        # Guess: last column
        train_only_cols = [train_df.columns[-1]]

    # Filter out likely non-target columns
    target_candidates = []
    for col in train_only_cols:
        nunique = train_df[col].nunique()
        if nunique <= 1:
            continue
        target_candidates.append(col)

    if len(target_candidates) == 1:
        meta["target_col"] = target_candidates[0]
        meta["target_cols"] = target_candidates
    elif len(target_candidates) > 1:
        # Multi-target or pick the best candidate
        # If all are binary (0/1), treat as multi-target
        all_binary = all(
            set(train_df[c].dropna().unique()).issubset({0, 1, 0.0, 1.0})
            for c in target_candidates
        )
        if all_binary:
            meta["target_col"] = target_candidates[0]
            meta["target_cols"] = target_candidates
        else:
            meta["target_col"] = target_candidates[0]
            meta["target_cols"] = [target_candidates[0]]

    # Detect ID column
    if test_df is not None:
        # First column of test that looks like an ID
        for col in test_df.columns:
            if col.lower() in ("id", "index", "row_id", "uid"):
                meta["id_col"] = col
                break
            if test_df[col].nunique() == len(test_df):
                meta["id_col"] = col
                break
        if not meta["id_col"] and len(test_df.columns) > 0:
            meta["id_col"] = test_df.columns[0]

    # Also check sample_submission for ID
    sample_sub = data_dir / "sample_submission.csv"
    if sample_sub.exists() and not meta["id_col"]:
        try:
            ss = pd.read_csv(sample_sub, nrows=5)
            meta["id_col"] = ss.columns[0]
        except Exception:
            pass

    # Profile target
    target_col = meta["target_col"]
    if target_col and target_col in train_df.columns:
        target_series = train_df[target_col]
        meta["target_nunique"] = int(target_series.nunique())
        meta["target_dtype"] = str(target_series.dtype)
        meta["target_is_bool"] = set(target_series.dropna().unique()).issubset(
            {True, False, 0, 1, 0.0, 1.0}
        )
        # Check if target looks like probabilities
        if target_series.dtype in ("float64", "float32"):
            if target_series.min() >= 0 and target_series.max() <= 1:
                meta["target_is_proba"] = True

    # Profile each column
    spend_cols = []
    binary_interaction_pairs = []
    binary_cols = []

    for col in train_df.columns:
        if col == target_col:
            continue
        cm: dict = {"role": "UNKNOWN"}
        series = train_df[col]
        nunique = int(series.nunique())
        null_pct = float(series.isnull().mean())

        # Detect role
        if col.lower() in ("id", "index", "row_id", "uid") or (
            nunique == len(train_df) and series.dtype == "object"
        ):
            cm["role"] = "ID"
        elif nunique <= 1:
            cm["role"] = "CONSTANT"
        elif series.dtype == "bool" or (
            set(series.dropna().unique()).issubset({True, False})
        ):
            cm["role"] = "BINARY_BOOL"
            binary_cols.append(col)
        elif series.dtype in ("int64", "float64") and nunique == 2:
            cm["role"] = "BINARY_NUMERIC"
            binary_cols.append(col)
        elif series.dtype == "object":
            # Check for bool strings
            vals = set(series.dropna().str.lower().unique())
            if vals.issubset({"true", "false", "yes", "no", "t", "f", "y", "n"}):
                cm["role"] = "BOOL_STR"
            elif nunique > HIGH_CARD_THRESHOLD:
                cm["role"] = "HIGH_CARD"
                # Check for name format
                if series.dropna().str.contains(r"[A-Z][a-z]+ [A-Z]", regex=True).mean() > 0.5:
                    cm["name_format"] = True
                # Check for structured string
                delim = _sniff_delimiter(series)
                if delim:
                    n_parts = int(series.dropna().str.split(re.escape(delim)).str.len().mode().iloc[0])
                    cm["role"] = "STRUCTURED_STR"
                    cm["structured_str"] = {"delimiter": delim, "n_parts": n_parts}
            else:
                cm["role"] = "CATEGORICAL"
        elif series.dtype in ("int64", "float64"):
            if nunique > HIGH_CARD_THRESHOLD:
                cm["role"] = "CONTINUOUS"
                # Detect spending columns
                if any(kw in col.lower() for kw in (
                    "spend", "price", "cost", "amount", "revenue", "service",
                    "food", "shopping", "spa", "vrdeck",
                )):
                    spend_cols.append(col)
            else:
                cm["role"] = "CATEGORICAL"

        cm["nunique"] = nunique
        cm["null_pct"] = null_pct
        meta["columns"][col] = cm

        report_lines.append(
            f"  [{col}]  dtype={series.dtype}  unique={nunique}  "
            f"null={series.isnull().sum()}({null_pct:.1%})  → {cm['role']}"
        )

    meta["spend_cols"] = spend_cols

    # Binary interaction pairs
    if len(binary_cols) >= 2:
        binary_interaction_pairs = [
            (binary_cols[i], binary_cols[j])
            for i in range(min(len(binary_cols), 3))
            for j in range(i + 1, min(len(binary_cols), 4))
        ]
    meta["binary_interaction_pairs"] = binary_interaction_pairs

    return "\n".join(report_lines), meta


def _read_resilient(path: Path):
    """Try multiple encodings; return None if all fail."""
    import pandas as pd
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, nrows=SAMPLE_ROWS, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
        except Exception:
            return None
    try:
        return pd.read_csv(
            path, nrows=SAMPLE_ROWS, encoding="utf-8", encoding_errors="replace",
            low_memory=False,
        )
    except Exception:
        return None


def _count_rows(path: Path) -> int:
    """Count data rows without loading the file."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except Exception:
        return -1


def _sniff_delimiter(series) -> str | None:
    """Return the most consistent delimiter in a string series."""
    sample = series.dropna().astype(str).head(200)
    if len(sample) < 10:
        return None
    for delim in DELIMITERS:
        counts = sample.str.count(re.escape(delim))
        if len(counts.mode()) == 0:
            continue
        mode_count = int(counts.mode().iloc[0])
        if mode_count > 0 and (counts == mode_count).mean() > 0.80:
            return delim
    return None
