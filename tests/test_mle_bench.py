"""Unit tests for MLE-Bench skill -- tests classification, EDA, and script generation."""

import base64
import io
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from classifier import classify_structural
from skills.mle_bench.eda import run_eda
from skills.mle_bench.modality import detect_modality
from skills.mle_bench.tabular import generate_tabular_script
from skills.mle_bench.runner import run_code


@pytest.fixture
def synthetic_tabular_competition(tmp_path):
    """Create a synthetic Kaggle-style tabular competition in tmp_path."""
    import numpy as np

    np.random.seed(42)
    n_train, n_test = 200, 50
    train = pd.DataFrame({
        "id": range(n_train),
        "feature_a": np.random.randn(n_train),
        "feature_b": np.random.choice(["cat", "dog", "bird"], n_train),
        "feature_c": np.random.randint(0, 100, n_train),
        "target": np.random.randint(0, 2, n_train),
    })
    test = pd.DataFrame({
        "id": range(n_train, n_train + n_test),
        "feature_a": np.random.randn(n_test),
        "feature_b": np.random.choice(["cat", "dog", "bird"], n_test),
        "feature_c": np.random.randint(0, 100, n_test),
    })
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    # description
    (tmp_path / "description.txt").write_text(
        "Binary classification: predict target (0 or 1). Metric: accuracy."
    )
    return tmp_path


@pytest.fixture
def synthetic_tar_bytes(synthetic_tabular_competition):
    """Create a tar.gz of the synthetic competition."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in synthetic_tabular_competition.iterdir():
            tar.add(f, arcname=f.name)
    return buf.getvalue()


# ===== Classifier tests =====

def test_classify_mle_bench():
    text = "You are participating in MLE-bench. Your goal is to produce a submission.csv."
    assert classify_structural(text) == "mle-bench"


def test_classify_mle_bench_no_match():
    text = "Write a hello world program"
    assert classify_structural(text) != "mle-bench"


# ===== EDA tests =====

def test_eda_basic(synthetic_tabular_competition):
    report, meta = run_eda(synthetic_tabular_competition)
    assert meta["train_file"] == "train.csv"
    assert meta["test_file"] == "test.csv"
    assert meta["target_col"] == "target"
    assert meta["id_col"] == "id"
    assert meta["n_train_rows"] == 200
    assert "target" not in meta["columns"]  # target excluded from features
    assert len(report) > 0


# ===== Modality tests =====

def test_modality_tabular(synthetic_tabular_competition):
    description = "Binary classification: predict target"
    modality = detect_modality(synthetic_tabular_competition, description)
    assert modality == "tabular"


# ===== Script generation tests =====

def test_generate_tabular_script(synthetic_tabular_competition):
    _, meta = run_eda(synthetic_tabular_competition)
    meta["competition_scoring"] = "accuracy"
    meta["competition_metric_label"] = "accuracy"
    script = generate_tabular_script(meta, str(synthetic_tabular_competition))
    assert script is not None
    assert "import lightgbm" in script
    assert "submission.csv" in script
    assert "train.csv" in script
    assert "test.csv" in script


# ===== Runner tests =====

@pytest.mark.asyncio
async def test_run_code_simple(synthetic_tabular_competition):
    script = f"""
import pandas as pd
import os
test = pd.read_csv(os.path.join('{synthetic_tabular_competition}', 'test.csv'))
sub = pd.DataFrame({{'id': test['id'], 'target': 0}})
sub.to_csv(os.path.join('{synthetic_tabular_competition}', 'submission.csv'), index=False)
print('done')
"""
    success, output = await run_code(script, synthetic_tabular_competition, timeout=30)
    assert success
    assert "done" in output
    assert (synthetic_tabular_competition / "submission.csv").exists()


# ===== End-to-end Phase 0 test =====

@pytest.mark.asyncio
async def test_phase0_end_to_end(synthetic_tabular_competition):
    """Test the full Phase 0 pipeline: EDA -> script gen -> execution."""
    _, meta = run_eda(synthetic_tabular_competition)
    meta["competition_scoring"] = "accuracy"
    meta["competition_metric_label"] = "accuracy"
    script = generate_tabular_script(meta, str(synthetic_tabular_competition))
    assert script is not None

    success, output = await run_code(script, synthetic_tabular_competition, timeout=120)
    assert success, f"Script failed:\n{output[-2000:]}"
    assert (synthetic_tabular_competition / "submission.csv").exists()

    sub = pd.read_csv(synthetic_tabular_competition / "submission.csv")
    assert "id" in sub.columns
    assert "target" in sub.columns
    assert len(sub) == 50  # n_test
