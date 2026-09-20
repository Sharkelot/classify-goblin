"""Reproducible optional training/evaluation utilities for local-distilbert.

The data and metric helpers are dependency-light.  PyTorch/Transformers are imported only
by the training and GPU-smoke commands, so the normal local API remains lightweight.
"""

from .data import (
    DEFAULT_LOCAL_TRACES,
    DecisionExample,
    load_examples,
    make_synthetic_cases,
    prepare_examples,
    read_jsonl,
    split_examples,
    write_jsonl,
)
from .metrics import compare_reports, evaluate_predictions, evaluate_workflow_precedence, fit_temperature

__all__ = [
    "DEFAULT_LOCAL_TRACES", "DecisionExample", "load_examples", "make_synthetic_cases",
    "prepare_examples", "read_jsonl", "split_examples", "write_jsonl",
    "compare_reports", "evaluate_predictions", "evaluate_workflow_precedence", "fit_temperature",
]
