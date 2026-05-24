"""Shared utilities for MLE-Bench skill."""

from __future__ import annotations

import io
import logging
import re
import tarfile
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_tar(data: bytes, dest: Path) -> None:
    """Extract a .tar.gz byte payload into dest."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        # Security: prevent path traversal
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in member.name:
                raise ValueError(f"Unsafe tar member: {member.name}")
        tar.extractall(dest)


def find_data_dir(work_dir: Path) -> Path:
    """Return the directory that contains description.md."""
    candidates = sorted(work_dir.rglob("description.md"))
    if candidates:
        return candidates[0].parent
    # Fallback: look for train.csv
    csv_candidates = sorted(work_dir.rglob("train.csv"))
    if csv_candidates:
        return csv_candidates[0].parent
    return work_dir


def read_description(work_dir: Path) -> str:
    """Return description.md text."""
    candidates = sorted(work_dir.rglob("description.md"))
    if candidates:
        return candidates[0].read_text(errors="replace")
    return "(no description.md found)"


def pre_extract_zips(data_dir: Path) -> None:
    """Extract zip files in data_dir for modality detection."""
    import zipfile
    for f in sorted(data_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() != ".zip":
            continue
        name_lower = f.name.lower()
        if not any(kw in name_lower for kw in ("train", "test", "data")):
            continue
        try:
            with zipfile.ZipFile(f, "r") as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                if not names:
                    continue
                first_parts = [n.split("/", 1) for n in names[:20]]
                common_prefix = None
                if all(len(p) == 2 for p in first_parts):
                    dirs = set(p[0] for p in first_parts)
                    if len(dirs) == 1:
                        common_prefix = next(iter(dirs))
                if common_prefix:
                    zf.extractall(data_dir)
                else:
                    if "train" in name_lower:
                        target = data_dir / "train"
                    elif "test" in name_lower:
                        target = data_dir / "test"
                    else:
                        target = data_dir / f.stem
                    target.mkdir(parents=True, exist_ok=True)
                    zf.extractall(target)
            logger.info("Pre-extracted %s (%d files)", f.name, len(names))
        except Exception as e:
            logger.warning("Failed to pre-extract %s: %s", f.name, e)


# Maps regex pattern → (sklearn_scoring_string, human_label)
_METRIC_PATTERNS: list[tuple[str, str, str]] = [
    (r"mean\s+column[- ]?wise\s+(ROC\s+)?AUC", "roc_auc", "mean_column_auc"),
    (r"log\s*loss|logarithmic\s+loss", "neg_log_loss", "log_loss"),
    (r"area\s+under\s+(the\s+)?ROC\s+curve|AUC[\s-]?ROC|ROC\s+AUC", "roc_auc", "roc_auc"),
    (r"\bAUC\b", "roc_auc", "roc_auc"),
    (r"root\s+mean\s+squared\s+error|\bRMSE\b", "neg_root_mean_squared_error", "rmse"),
    (r"mean\s+squared\s+error|\bMSE\b", "neg_mean_squared_error", "mse"),
    (r"mean\s+absolute\s+error|\bMAE\b", "neg_mean_absolute_error", "mae"),
    (r"\bRMSLE\b|root\s+mean\s+squared\s+log", "neg_root_mean_squared_log_error", "rmsle"),
    (r"classification\s+accuracy|\baccuracy\b", "accuracy", "accuracy"),
    (r"\bF1[\s-]?score\b|\bF1\b", "f1", "f1"),
    (r"Matthews\s+Correlation|MCC", "matthews_corrcoef", "mcc"),
    (r"Cohen.?s?\s+Kappa|\bkappa\b", "cohen_kappa_score", "kappa"),
    (r"\bR\s*²\b|\bR-?squared\b|\bR2\b", "r2", "r2"),
]


def detect_competition_metric(description: str, eda_meta: dict) -> tuple[str, str]:
    """Detect the competition's evaluation metric.

    Returns (sklearn_scoring_string, human_label).
    """
    if description and description != "(no description.md found)":
        desc_lower = description.lower()
        eval_section = description
        for marker in ("## evaluation", "## metric", "### metric", "### evaluation",
                       "evaluation\n", "metric\n"):
            idx = desc_lower.find(marker)
            if idx >= 0:
                eval_section = description[idx:idx + 1500]
                break

        for pattern, sklearn_name, label in _METRIC_PATTERNS:
            if re.search(pattern, eval_section, re.IGNORECASE):
                return sklearn_name, label

    # EDA-based inference
    target_is_proba = eda_meta.get("target_is_proba", False)
    target_is_bool = eda_meta.get("target_is_bool", False)
    target_nunique = eda_meta.get("target_nunique", 0)
    target_dtype = eda_meta.get("target_dtype", "")
    columns = eda_meta.get("columns", {})
    target_col = eda_meta.get("target_col", "")
    target_cols = eda_meta.get("target_cols", [])

    is_binary = target_is_bool or (
        columns.get(target_col, {}).get("role") in ("BINARY_BOOL", "BINARY_NUMERIC")
    ) or (target_nunique == 2)
    is_multi_target = len(target_cols) > 1

    if is_multi_target:
        return "roc_auc", "roc_auc"
    if target_is_proba:
        return "roc_auc", "roc_auc"
    if is_binary:
        return "accuracy", "accuracy"
    if not is_binary and target_nunique > 2 and (
        target_dtype.startswith("int") or target_dtype == "object"
    ):
        return "accuracy", "accuracy"
    return "neg_root_mean_squared_error", "rmse"
