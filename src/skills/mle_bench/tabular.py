"""Tabular Phase 0 -- deterministic ML script generator (zero LLM calls).

Generates a complete Python script from EDA metadata that handles:
- Data loading and target/ID detection
- Boolean string conversion
- Structured string splitting
- Categorical encoding (label + frequency)
- Missing value imputation
- Spend/interaction features
- LightGBM + CatBoost ensemble
- Cross-validation scoring
- Submission generation
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def generate_tabular_script(meta: dict, data_dir: str) -> str | None:
    """Generate a Phase 0 tabular ML script from EDA metadata.

    Returns a runnable Python script string, or None if metadata is insufficient.
    """
    train_file = meta.get("train_file")
    test_file = meta.get("test_file")
    target_col = meta.get("target_col")
    id_col = meta.get("id_col")
    columns = meta.get("columns", {})

    if not all([train_file, test_file, target_col, id_col]):
        return None

    target_is_bool = meta.get("target_is_bool", False)
    target_is_proba = meta.get("target_is_proba", False)
    target_cols = meta.get("target_cols", [])
    target_nunique = meta.get("target_nunique", 0)
    target_dtype = meta.get("target_dtype", "")
    spend_cols = meta.get("spend_cols", [])
    interaction_pairs = meta.get("binary_interaction_pairs", [])
    n_train = meta.get("n_train_rows", 0)

    # Determine task type
    is_binary = target_is_bool or (
        columns.get(target_col, {}).get("role") in ("BINARY_BOOL", "BINARY_NUMERIC")
    ) or (target_nunique == 2)
    is_multiclass = (not is_binary and target_nunique > 2
                     and (target_dtype.startswith("int") or target_dtype == "object"))
    is_multi_target = len(target_cols) > 1

    # Classify columns by role
    bool_str_cols = []
    structured_str_cols = []
    categorical_cols = []
    continuous_cols = []
    high_card_cols = []
    drop_cols = []

    for col_name, cm in columns.items():
        role = cm.get("role", "UNKNOWN")
        if col_name == target_col or col_name in target_cols:
            continue
        if role == "ID":
            continue
        elif role == "BOOL_STR":
            bool_str_cols.append(col_name)
        elif role == "STRUCTURED_STR" and "structured_str" in cm:
            structured_str_cols.append({
                "name": col_name,
                "delimiter": cm["structured_str"]["delimiter"],
                "n_parts": cm["structured_str"]["n_parts"],
            })
        elif role in ("CATEGORICAL", "ORDINAL"):
            categorical_cols.append(col_name)
        elif role == "CONTINUOUS":
            continuous_cols.append(col_name)
        elif role == "HIGH_CARD":
            high_card_cols.append(col_name)
        elif role == "CONSTANT":
            drop_cols.append(col_name)
        elif role in ("BINARY_BOOL", "BINARY_NUMERIC"):
            pass  # keep as-is

    # Determine scoring metric
    comp_scoring = meta.get("competition_scoring", "accuracy")
    comp_metric = meta.get("competition_metric_label", "accuracy")

    # Build script
    lines: list[str] = []
    w = lines.append

    w("import os")
    w("import sys")
    w("import warnings")
    w("import time")
    w("import gc")
    w("warnings.filterwarnings('ignore')")
    w("import numpy as np")
    w("import pandas as pd")
    w("from sklearn.model_selection import StratifiedKFold, KFold")
    w("from sklearn.preprocessing import LabelEncoder")
    w("from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score")
    w("")
    w("START = time.time()")
    w(f"DATA_DIR = '{data_dir}'")
    w("")
    w("# ===== Load Data =====")
    w("print('Loading data...', flush=True)")
    w(f"train = pd.read_csv(os.path.join(DATA_DIR, '{train_file}'))")
    w(f"test = pd.read_csv(os.path.join(DATA_DIR, '{test_file}'))")
    w(f"print(f'Train: {{train.shape}}, Test: {{test.shape}}', flush=True)")
    w("")

    # Memory management for large datasets
    if n_train > 500_000:
        w(f"# Subsample for speed (original: {n_train} rows)")
        w(f"if len(train) > 500000:")
        w(f"    train = train.sample(500000, random_state=42).reset_index(drop=True)")
        w(f"    print(f'Subsampled to {{len(train)}} rows', flush=True)")
        w("")

    w(f"TARGET_COL = '{target_col}'")
    w(f"ID_COL = '{id_col}'")
    if is_multi_target:
        w(f"TARGET_COLS = {target_cols}")
    w("")

    # Extract target
    if is_multi_target:
        w("y_all = train[TARGET_COLS].values")
    else:
        w("y = train[TARGET_COL].values")
        if is_multiclass and target_dtype == "object":
            w("# Encode string target")
            w("target_le = LabelEncoder()")
            w("y = target_le.fit_transform(y)")

    w("train_ids = train[ID_COL] if ID_COL in train.columns else None")
    w("test_ids = test[ID_COL]")
    w("")

    # Drop target and ID from features
    w("# Prepare feature dataframes")
    if is_multi_target:
        w("drop_cols_list = TARGET_COLS + [ID_COL]")
    else:
        w("drop_cols_list = [TARGET_COL, ID_COL]")
    if drop_cols:
        w(f"drop_cols_list += {drop_cols}")
    w("train_feats = train.drop(columns=[c for c in drop_cols_list if c in train.columns])")
    w("test_feats = test.drop(columns=[c for c in drop_cols_list if c in test.columns])")
    w("# Align columns")
    w("common_cols = [c for c in train_feats.columns if c in test_feats.columns]")
    w("train_feats = train_feats[common_cols]")
    w("test_feats = test_feats[common_cols]")
    w("")

    # Boolean string conversion
    if bool_str_cols:
        w("# Convert boolean strings")
        w("bool_map = {'true': 1, 'false': 0, 'yes': 1, 'no': 0, 't': 1, 'f': 0, 'y': 1, 'n': 0}")
        for col in bool_str_cols:
            w(f"for df in [train_feats, test_feats]:")
            w(f"    if '{col}' in df.columns:")
            w(f"        df['{col}'] = df['{col}'].astype(str).str.lower().map(bool_map)")
        w("")

    # Structured string splitting
    if structured_str_cols:
        w("# Split structured strings")
        for ssc in structured_str_cols:
            col = ssc["name"]
            delim = ssc["delimiter"]
            n_parts = ssc["n_parts"]
            w(f"for df in [train_feats, test_feats]:")
            w(f"    if '{col}' in df.columns:")
            w(f"        parts = df['{col}'].astype(str).str.split('{delim}', expand=True)")
            w(f"        for i in range(min({n_parts}, parts.shape[1])):")
            w(f"            part_col = f'{col}_part_{{i}}'")
            w(f"            df[part_col] = parts[i]")
            w(f"            # Try to convert to numeric")
            w(f"            numeric = pd.to_numeric(df[part_col], errors='coerce')")
            w(f"            if numeric.notna().mean() > 0.8:")
            w(f"                df[part_col] = numeric")
            w(f"        df.drop(columns=['{col}'], inplace=True, errors='ignore')")
        w("")

    # Spend aggregation features
    if spend_cols:
        w("# Spend aggregation features")
        spend_in_data = [c for c in spend_cols if c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"] or True]
        w(f"spend_cols_present = [c for c in {spend_cols} if c in train_feats.columns]")
        w("if spend_cols_present:")
        w("    for df in [train_feats, test_feats]:")
        w("        df['TotalSpend'] = df[spend_cols_present].fillna(0).sum(axis=1)")
        w("        df['IsZeroSpend'] = (df['TotalSpend'] == 0).astype(int)")
        w("        df['MeanSpend'] = df[spend_cols_present].fillna(0).mean(axis=1)")
        w("        for c in spend_cols_present:")
        w("            df[f'{c}_IsZero'] = (df[c].fillna(0) == 0).astype(int)")
        w("")

    # Interaction features
    if interaction_pairs:
        w("# Binary interaction features")
        for col_a, col_b in interaction_pairs[:5]:
            w(f"for df in [train_feats, test_feats]:")
            w(f"    if '{col_a}' in df.columns and '{col_b}' in df.columns:")
            w(f"        df['{col_a}_x_{col_b}'] = df['{col_a}'].fillna(0).astype(float) * df['{col_b}'].fillna(0).astype(float)")
        w("")

    # Encode categoricals
    w("# Encode categorical columns")
    w("cat_cols = train_feats.select_dtypes(include=['object', 'category']).columns.tolist()")
    w("label_encoders = {}")
    w("for col in cat_cols:")
    w("    le = LabelEncoder()")
    w("    combined = pd.concat([train_feats[col], test_feats[col]], axis=0).astype(str).fillna('_missing_')")
    w("    le.fit(combined)")
    w("    train_feats[col] = le.transform(train_feats[col].astype(str).fillna('_missing_'))")
    w("    test_feats[col] = le.transform(test_feats[col].astype(str).fillna('_missing_'))")
    w("    label_encoders[col] = le")
    w("")

    # Convert to numeric
    w("# Ensure all numeric")
    w("train_feats = train_feats.apply(pd.to_numeric, errors='coerce')")
    w("test_feats = test_feats.apply(pd.to_numeric, errors='coerce')")
    w("")

    # Handle NaN
    w("# Fill remaining NaN")
    w("train_feats = train_feats.fillna(-999)")
    w("test_feats = test_feats.fillna(-999)")
    w("")

    w(f"print(f'Features: {{train_feats.shape[1]}} ({{time.time()-START:.0f}}s)', flush=True)")
    w("")

    # Model training
    w("# ===== Model Training =====")
    w("try:")
    w("    import lightgbm as lgb")
    w("    HAS_LGBM = True")
    w("except ImportError:")
    w("    HAS_LGBM = False")
    w("")
    w("try:")
    w("    from catboost import CatBoostClassifier, CatBoostRegressor")
    w("    HAS_CATBOOST = True")
    w("except ImportError:")
    w("    HAS_CATBOOST = False")
    w("")

    if is_multi_target:
        _generate_multi_target(w, target_cols, comp_scoring)
    elif is_binary:
        _generate_binary(w, comp_scoring, comp_metric)
    elif is_multiclass:
        _generate_multiclass(w, target_nunique, target_dtype, comp_scoring)
    else:
        _generate_regression(w, comp_scoring, comp_metric)

    # Submission
    w("")
    w("# ===== Write Submission =====")
    w("submission = pd.DataFrame()")
    w("submission[ID_COL] = test_ids")
    if is_multi_target:
        w("for i, tc in enumerate(TARGET_COLS):")
        w("    submission[tc] = test_preds[:, i]")
    elif is_multiclass and target_dtype == "object":
        w("submission[TARGET_COL] = target_le.inverse_transform(test_preds.astype(int))")
    else:
        w("submission[TARGET_COL] = test_preds")
    w("")
    w("submission.to_csv(os.path.join(DATA_DIR, 'submission.csv'), index=False)")
    w("print(f'\\nSubmission: {submission.shape}', flush=True)")
    w("print(submission.head(), flush=True)")
    w(f"print(f'Total time: {{time.time()-START:.0f}}s', flush=True)")

    return "\n".join(lines)


def _generate_binary(w, comp_scoring: str, comp_metric: str):
    """Generate binary classification training code."""
    predict_proba = comp_scoring in ("roc_auc", "neg_log_loss")

    w("# Binary classification ensemble")
    w("N_FOLDS = 5")
    w("X = train_feats.values")
    w("X_test = test_feats.values")
    w("")
    w("oof_preds = np.zeros(len(X))")
    w("test_preds = np.zeros(len(X_test))")
    w("")
    w("if HAS_LGBM:")
    w("    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)")
    w("    lgb_params = {")
    w("        'objective': 'binary',")
    w("        'metric': 'binary_logloss',")
    w("        'verbosity': -1,")
    w("        'n_estimators': 1000,")
    w("        'learning_rate': 0.05,")
    w("        'num_leaves': 63,")
    w("        'max_depth': -1,")
    w("        'min_child_samples': 20,")
    w("        'subsample': 0.8,")
    w("        'colsample_bytree': 0.8,")
    w("        'random_state': 42,")
    w("        'n_jobs': -1,")
    w("    }")
    w("")
    w("    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):")
    w("        X_tr, X_val = X[tr_idx], X[val_idx]")
    w("        y_tr, y_val = y[tr_idx], y[val_idx]")
    w("        model = lgb.LGBMClassifier(**lgb_params)")
    w("        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],")
    w("                  callbacks=[lgb.early_stopping(50, verbose=False)])")
    if predict_proba:
        w("        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]")
        w("        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS")
    else:
        w("        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]")
        w("        test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS")
    w("")
    if predict_proba:
        w("    cv_score = roc_auc_score(y, oof_preds)")
    else:
        w("    cv_score = accuracy_score(y, (oof_preds > 0.5).astype(int))")
    w(f"    print(f'LGBM CV {comp_metric}: {{cv_score:.5f}}', flush=True)")
    w("")
    w("    # CatBoost ensemble member")
    w("    if HAS_CATBOOST:")
    w("        cb_test_preds = np.zeros(len(X_test))")
    w("        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):")
    w("            cb = CatBoostClassifier(")
    w("                iterations=500, learning_rate=0.05, depth=6,")
    w("                verbose=0, random_seed=42, task_type='CPU')")
    w("            cb.fit(X[tr_idx], y[tr_idx], eval_set=(X[val_idx], y[val_idx]),")
    w("                   early_stopping_rounds=50, verbose=0)")
    w("            cb_test_preds += cb.predict_proba(X_test)[:, 1] / N_FOLDS")
    w("        # Blend LGBM + CatBoost")
    w("        test_preds = 0.6 * test_preds + 0.4 * cb_test_preds")
    w("        print('Blended LGBM + CatBoost', flush=True)")
    w("")
    if not predict_proba:
        w("    test_preds = (test_preds > 0.5).astype(int)")
    w("")
    w("else:")
    w("    # Fallback: sklearn GradientBoosting")
    w("    from sklearn.ensemble import GradientBoostingClassifier")
    w("    model = GradientBoostingClassifier(n_estimators=200, random_state=42)")
    w("    model.fit(X, y)")
    if predict_proba:
        w("    test_preds = model.predict_proba(X_test)[:, 1]")
    else:
        w("    test_preds = model.predict(X_test)")
    w("")
    w(f"print(f'CV_METRIC={comp_metric}', flush=True)")
    w(f"print(f'CV_SCORE={{cv_score if \"cv_score\" in dir() else \"SKIPPED\"}}', flush=True)")


def _generate_multiclass(w, target_nunique: int, target_dtype: str, comp_scoring: str):
    """Generate multiclass classification training code."""
    w(f"# Multiclass classification ({target_nunique} classes)")
    w("N_FOLDS = 5")
    w("X = train_feats.values")
    w("X_test = test_feats.values")
    w(f"N_CLASSES = {target_nunique}")
    w("")
    w("test_preds = np.zeros(len(X_test))")
    w("")
    w("if HAS_LGBM:")
    w("    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)")
    w("    lgb_params = {")
    w("        'objective': 'multiclass',")
    w(f"        'num_class': N_CLASSES,")
    w("        'metric': 'multi_logloss',")
    w("        'verbosity': -1,")
    w("        'n_estimators': 800,")
    w("        'learning_rate': 0.05,")
    w("        'num_leaves': 63,")
    w("        'subsample': 0.8,")
    w("        'colsample_bytree': 0.8,")
    w("        'random_state': 42,")
    w("        'n_jobs': -1,")
    w("    }")
    w("")
    w("    oof_preds = np.zeros((len(X), N_CLASSES))")
    w("    test_preds_proba = np.zeros((len(X_test), N_CLASSES))")
    w("")
    w("    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):")
    w("        model = lgb.LGBMClassifier(**lgb_params)")
    w("        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])],")
    w("                  callbacks=[lgb.early_stopping(50, verbose=False)])")
    w("        oof_preds[val_idx] = model.predict_proba(X[val_idx])")
    w("        test_preds_proba += model.predict_proba(X_test) / N_FOLDS")
    w("")
    w("    test_preds = test_preds_proba.argmax(axis=1)")
    w("    cv_score = accuracy_score(y, oof_preds.argmax(axis=1))")
    w("    print(f'LGBM CV accuracy: {cv_score:.5f}', flush=True)")
    w("")
    w("else:")
    w("    from sklearn.ensemble import GradientBoostingClassifier")
    w("    model = GradientBoostingClassifier(n_estimators=200, random_state=42)")
    w("    model.fit(X, y)")
    w("    test_preds = model.predict(X_test)")
    w("")
    w("print(f'CV_METRIC=accuracy', flush=True)")
    w("print(f'CV_SCORE={cv_score if \"cv_score\" in dir() else \"SKIPPED\"}', flush=True)")


def _generate_regression(w, comp_scoring: str, comp_metric: str):
    """Generate regression training code."""
    use_log = comp_metric in ("rmsle",)

    w("# Regression")
    w("N_FOLDS = 5")
    w("X = train_feats.values")
    w("X_test = test_feats.values")
    w("")
    if use_log:
        w("# Log-transform target for RMSLE")
        w("y_orig = y.copy()")
        w("y = np.log1p(y)")
    w("")
    w("test_preds = np.zeros(len(X_test))")
    w("")
    w("if HAS_LGBM:")
    w("    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)")
    w("    lgb_params = {")
    w("        'objective': 'regression',")
    w("        'metric': 'rmse',")
    w("        'verbosity': -1,")
    w("        'n_estimators': 1000,")
    w("        'learning_rate': 0.05,")
    w("        'num_leaves': 63,")
    w("        'subsample': 0.8,")
    w("        'colsample_bytree': 0.8,")
    w("        'random_state': 42,")
    w("        'n_jobs': -1,")
    w("    }")
    w("")
    w("    oof_preds = np.zeros(len(X))")
    w("    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):")
    w("        model = lgb.LGBMRegressor(**lgb_params)")
    w("        model.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])],")
    w("                  callbacks=[lgb.early_stopping(50, verbose=False)])")
    w("        oof_preds[val_idx] = model.predict(X[val_idx])")
    w("        test_preds += model.predict(X_test) / N_FOLDS")
    w("")
    if use_log:
        w("    oof_preds = np.expm1(oof_preds)")
        w("    test_preds = np.expm1(test_preds)")
        w("    y_eval = y_orig")
    else:
        w("    y_eval = y")
    w("    rmse = np.sqrt(mean_squared_error(y_eval, oof_preds))")
    w("    r2 = r2_score(y_eval, oof_preds)")
    w(f"    print(f'LGBM CV RMSE: {{rmse:.5f}}, R2: {{r2:.5f}}', flush=True)")
    w("    cv_score = r2")
    w("")
    w("else:")
    w("    from sklearn.ensemble import GradientBoostingRegressor")
    w("    model = GradientBoostingRegressor(n_estimators=200, random_state=42)")
    w("    model.fit(X, y)")
    w("    test_preds = model.predict(X_test)")
    if use_log:
        w("    test_preds = np.expm1(test_preds)")
    w("")
    w(f"print(f'CV_METRIC={comp_metric}', flush=True)")
    w("print(f'CV_SCORE={cv_score if \"cv_score\" in dir() else \"SKIPPED\"}', flush=True)")


def _generate_multi_target(w, target_cols: list[str], comp_scoring: str):
    """Generate multi-target prediction code."""
    w(f"# Multi-target prediction ({len(target_cols)} targets)")
    w("X = train_feats.values")
    w("X_test = test_feats.values")
    w(f"test_preds = np.zeros((len(X_test), {len(target_cols)}))")
    w("")
    w("if HAS_LGBM:")
    w("    lgb_params = {")
    w("        'objective': 'binary',")
    w("        'metric': 'binary_logloss',")
    w("        'verbosity': -1,")
    w("        'n_estimators': 500,")
    w("        'learning_rate': 0.05,")
    w("        'num_leaves': 31,")
    w("        'random_state': 42,")
    w("        'n_jobs': -1,")
    w("    }")
    w("")
    w("    for i, tc in enumerate(TARGET_COLS):")
    w("        y_t = y_all[:, i]")
    w("        model = lgb.LGBMClassifier(**lgb_params)")
    w("        model.fit(X, y_t)")
    w("        test_preds[:, i] = model.predict_proba(X_test)[:, 1]")
    w("        print(f'  Target {tc}: done', flush=True)")
    w("")
    w("else:")
    w("    from sklearn.ensemble import GradientBoostingClassifier")
    w("    for i, tc in enumerate(TARGET_COLS):")
    w("        y_t = y_all[:, i]")
    w("        model = GradientBoostingClassifier(n_estimators=100, random_state=42)")
    w("        model.fit(X, y_t)")
    w("        test_preds[:, i] = model.predict_proba(X_test)[:, 1]")
