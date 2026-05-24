"""Modality detection for MLE-Bench competitions."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_modality(data_dir: Path, description: str) -> str:
    """Detect competition modality from data files and description.

    Returns one of: "tabular", "text", "image", "image_regression", "audio"
    """
    desc_lower = description.lower()
    files = [p.name for p in data_dir.iterdir()] if data_dir.exists() else []
    files_lower = [f.lower() for f in files]

    # Audio
    audio_exts = (".aif", ".wav", ".mp3", ".flac", ".ogg")
    has_audio_files = any(f.endswith(audio_exts) for f in files_lower)
    if not has_audio_files:
        for d in data_dir.iterdir():
            if d.is_dir():
                for p in list(d.iterdir())[:20]:
                    if p.is_file() and p.suffix.lower() in audio_exts:
                        has_audio_files = True
                        break
            if has_audio_files:
                break
    if has_audio_files:
        logger.info("Modality: audio")
        return "audio"

    # Image regression (pixel-level submission)
    sample_sub = _find_sample_submission(data_dir)
    if sample_sub and sample_sub.exists():
        try:
            import pandas as pd
            samp = pd.read_csv(sample_sub, nrows=5)
            if len(samp.columns) == 2:
                id_col = samp.columns[0]
                sample_ids = samp[id_col].astype(str).tolist()
                if all(
                    id_.count("_") >= 2 and all(p.isdigit() for p in id_.split("_"))
                    for id_ in sample_ids[:5]
                ):
                    logger.info("Modality: image_regression")
                    return "image_regression"
        except Exception:
            pass

    # Image
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    has_image_dirs = any(
        d.is_dir() and any(
            p.suffix.lower() in image_exts
            for p in list(d.iterdir())[:10] if p.is_file()
        )
        for d in data_dir.iterdir() if d.is_dir()
    )
    has_image_zips = any(
        f.endswith(".zip") and any(kw in f for kw in ("train", "test", "image"))
        for f in files_lower
    )
    image_keywords = ("image", "photo", "picture", "pixel", "aerial", "visual",
                      "cnn", "convolutional", "dogs", "cats", "cactus")
    desc_has_image = any(kw in desc_lower for kw in image_keywords)

    if (has_image_zips or has_image_dirs) and desc_has_image:
        logger.info("Modality: image")
        return "image"

    # Text/NLP
    train_path = data_dir / "train.csv"
    if train_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(train_path, nrows=50)
            text_cols = []
            for col in df.select_dtypes(include=["object"]).columns:
                avg_len = df[col].dropna().str.len().mean()
                if avg_len > 100:
                    text_cols.append(col)
            n_other = len(df.columns) - len(text_cols) - 1
            if text_cols and n_other <= 8:
                nlp_keywords = ("toxic", "comment", "text", "review", "sentiment",
                                "spam", "hate", "offensive", "nlp", "language")
                if any(kw in desc_lower for kw in nlp_keywords) or text_cols:
                    logger.info("Modality: text (columns: %s)", text_cols)
                    return "text"
        except Exception:
            pass

    logger.info("Modality: tabular (default)")
    return "tabular"


def _find_sample_submission(data_dir: Path) -> Path | None:
    """Find the sample submission CSV."""
    for candidate in ("sample_submission.csv", "sampleSubmission.csv", "SampleSubmission.csv"):
        if (data_dir / candidate).exists():
            return data_dir / candidate
    for f in data_dir.iterdir():
        if f.suffix == ".csv" and "sub" in f.name.lower():
            return f
    return None
