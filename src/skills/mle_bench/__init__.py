"""MLE-Bench skill -- Kaggle competition solver via deterministic ML pipelines.

Ported from other-agents/mle-bench-green/src/purple/server.py.
Routes competitions to specialized pipelines based on data modality:
  - Tabular: Phase 0 deterministic LGBM/CatBoost ensemble (zero LLM calls)
  - Text/NLP: NB-SVM with TF-IDF word+char features
  - Image: Simple CNN baseline
  - Fallback: LLM-generated code

The skill receives the full A2A Message (text instructions + competition.tar.gz),
extracts data, runs EDA, and returns submission.csv content.
"""

from .solve import solve_mle_bench

__all__ = ["solve_mle_bench"]
