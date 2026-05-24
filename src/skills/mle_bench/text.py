"""Text/NLP pipeline -- NB-SVM with TF-IDF for text classification.

Generates a self-contained Python script that:
- Builds word TF-IDF (50K features, 1-3 ngrams)
- Builds char TF-IDF (30K features, 2-5 char_wb) if time allows
- Applies NB-SVM weighting (Wang & Manning 2012)
- Trains LR at multiple C values and blends
- Produces submission.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_text_script(data_dir: Path, description: str) -> str | None:
    """Generate NB-SVM text pipeline script. Returns script string or None."""
    import pandas as pd

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_sub = None
    for name in ("sample_submission.csv", "sampleSubmission.csv", "SampleSubmission.csv"):
        if (data_dir / name).exists():
            sample_sub = data_dir / name
            break

    if not train_path.exists():
        return None

    df = pd.read_csv(train_path, nrows=100)
    samp = pd.read_csv(sample_sub, nrows=5) if sample_sub else None

    # Detect text column
    text_col = None
    max_avg_len = 0
    for col in df.select_dtypes(include=["object"]).columns:
        avg_len = df[col].dropna().str.len().mean()
        if avg_len > max_avg_len:
            max_avg_len = avg_len
            text_col = col

    if text_col is None:
        return None

    # Detect ID column
    id_col = df.columns[0]
    for col in df.columns:
        if col.lower() == "id":
            id_col = col
            break

    # Detect target columns
    if samp is not None:
        target_cols = [c for c in samp.columns if c != id_col]
    else:
        target_cols = [
            c for c in df.columns
            if c != text_col and c != id_col
            and df[c].dtype in ("int64", "float64")
        ]

    is_multi_label = len(target_cols) > 1

    # Determine if we need probabilities
    desc_lower = description.lower()
    predict_proba = "roc" in desc_lower or "auc" in desc_lower or "log_loss" in desc_lower

    lines: list[str] = []
    w = lines.append

    w("import pandas as pd")
    w("import numpy as np")
    w("import time")
    w("import os")
    w("import sys")
    w("import warnings")
    w("warnings.filterwarnings('ignore')")
    w("from sklearn.feature_extraction.text import TfidfVectorizer")
    w("from sklearn.linear_model import LogisticRegression")
    w("from scipy.sparse import csr_matrix")
    w("")
    w("START_TIME = time.time()")
    w("TIME_BUDGET = 600")
    w("")
    w("def elapsed(): return time.time() - START_TIME")
    w("def time_left(): return TIME_BUDGET - elapsed()")
    w("")
    w("def pr(y, x, alpha=1.0):")
    w("    '''NB log-ratios (Wang & Manning 2012).'''")
    w("    p = alpha + x[y == 1].sum(axis=0)")
    w("    q = alpha + x[y == 0].sum(axis=0)")
    w("    r = np.log(p / p.sum()) - np.log(q / q.sum())")
    w("    return np.asarray(r).flatten()")
    w("")
    w(f"DATA_DIR = '{data_dir}'")
    w("print('Loading data...', flush=True)")
    w(f"train = pd.read_csv(os.path.join(DATA_DIR, '{train_path.name}'))")
    w(f"test = pd.read_csv(os.path.join(DATA_DIR, '{test_path.name}'))")
    w("")
    w(f"TEXT_COL = '{text_col}'")
    w(f"ID_COL = '{id_col}'")
    w(f"TARGET_COLS = {target_cols}")
    w("")
    w("print(f'Train: {train.shape}, Test: {test.shape} ({elapsed():.0f}s)', flush=True)")
    w("train[TEXT_COL] = train[TEXT_COL].fillna('')")
    w("test[TEXT_COL] = test[TEXT_COL].fillna('')")
    w("")
    w("all_text = pd.concat([train[TEXT_COL], test[TEXT_COL]], axis=0)")
    w("n_train = len(train)")
    w("all_text = all_text.str.replace(r'\\n+', ' ', regex=True)")
    w("all_text = all_text.str.replace(r'\\s+', ' ', regex=True).str.strip()")
    w("")
    w("# Word TF-IDF")
    w("print('Fitting word TF-IDF (50K, 1-3 ngrams)...', flush=True)")
    w("tfidf_word = TfidfVectorizer(")
    w("    max_features=50000, strip_accents='unicode', analyzer='word',")
    w("    ngram_range=(1, 3), sublinear_tf=True, dtype=np.float32, min_df=2)")
    w("X_all_word = tfidf_word.fit_transform(all_text)")
    w("X_train_word = X_all_word[:n_train]")
    w("X_test_word = X_all_word[n_train:]")
    w("del X_all_word")
    w("print(f'  Word features: {X_train_word.shape[1]} ({elapsed():.0f}s)', flush=True)")
    w("")
    w("# Char TF-IDF (time-gated, reduced params for speed)")
    w("X_train_char = None")
    w("X_test_char = None")
    w("if time_left() > 200:")
    w("    print('Fitting char TF-IDF (30K, 2-5 char_wb)...', flush=True)")
    w("    tfidf_char = TfidfVectorizer(")
    w("        max_features=30000, strip_accents='unicode', analyzer='char_wb',")
    w("        ngram_range=(2, 5), sublinear_tf=True, dtype=np.float32, min_df=2)")
    w("    X_all_char = tfidf_char.fit_transform(all_text)")
    w("    X_train_char = X_all_char[:n_train]")
    w("    X_test_char = X_all_char[n_train:]")
    w("    del X_all_char")
    w("    print(f'  Char features: {X_train_char.shape[1]} ({elapsed():.0f}s)', flush=True)")
    w("else:")
    w("    print('  Skipping char TF-IDF (time budget)', flush=True)")
    w("")
    w("del all_text")
    w("")
    w(f"submission = test[[ID_COL]].copy()")
    w("")

    if is_multi_label:
        w("for target_idx, target in enumerate(TARGET_COLS):")
        w("    print(f'\\n=== Target: {target} ({target_idx+1}/{len(TARGET_COLS)}) ===', flush=True)")
        w("    y = train[target].values")
        w("")
        w("    # NB-SVM word features, multi-C blend")
        w("    r_word = pr(y, X_train_word)")
        w("    X_nb_train_word = X_train_word.multiply(r_word).tocsr()")
        w("    X_nb_test_word = X_test_word.multiply(r_word).tocsr()")
        w("")
        w("    word_preds = []")
        w("    for C_val in [2.0, 4.0, 8.0]:")
        w("        lr = LogisticRegression(C=C_val, solver='liblinear', max_iter=300, random_state=42)")
        w("        lr.fit(X_nb_train_word, y)")
        w("        word_preds.append(lr.predict_proba(X_nb_test_word)[:, 1])")
        w("    preds_word = np.mean(word_preds, axis=0)")
        w("")
        w("    if X_train_char is not None:")
        w("        r_char = pr(y, X_train_char)")
        w("        X_nb_train_char = X_train_char.multiply(r_char).tocsr()")
        w("        X_nb_test_char = X_test_char.multiply(r_char).tocsr()")
        w("        char_preds = []")
        w("        for C_val in [2.0, 4.0, 8.0]:")
        w("            lr = LogisticRegression(C=C_val, solver='liblinear', max_iter=300, random_state=42)")
        w("            lr.fit(X_nb_train_char, y)")
        w("            char_preds.append(lr.predict_proba(X_nb_test_char)[:, 1])")
        w("        preds_char = np.mean(char_preds, axis=0)")
        w("        submission[target] = 0.55 * preds_word + 0.45 * preds_char")
        w("    else:")
        w("        submission[target] = preds_word")
        w("")
        w("    print(f'    Done ({time_left():.0f}s left)', flush=True)")
    else:
        target = target_cols[0]
        w(f"y = train['{target}'].values")
        w("r_word = pr(y, X_train_word)")
        w("X_nb_train_word = X_train_word.multiply(r_word).tocsr()")
        w("X_nb_test_word = X_test_word.multiply(r_word).tocsr()")
        w("")
        w("word_preds = []")
        w("for C_val in [2.0, 4.0, 8.0]:")
        w("    lr = LogisticRegression(C=C_val, solver='liblinear', max_iter=300, random_state=42)")
        w("    lr.fit(X_nb_train_word, y)")
        if predict_proba:
            w("    word_preds.append(lr.predict_proba(X_nb_test_word)[:, 1])")
        else:
            w("    word_preds.append(lr.predict(X_nb_test_word))")
        if predict_proba:
            w(f"submission['{target}'] = np.mean(word_preds, axis=0)")
        else:
            w(f"from scipy.stats import mode as scipy_mode")
            w(f"submission['{target}'] = scipy_mode(np.array(word_preds), axis=0).mode.flatten()")

    w("")
    w("submission.to_csv(os.path.join(DATA_DIR, 'submission.csv'), index=False)")
    w("print(f'\\nSubmission: {submission.shape}', flush=True)")
    w("print(f'Total time: {elapsed():.0f}s', flush=True)")

    return "\n".join(lines)
