"""Held-out temperature calibration for a trained local-distilbert checkpoint.

This is a calibration pass, not a retrain: it loads an existing checkpoint,
fits a scalar temperature on a held-out split (one that the trainer never saw),
and records the before/after metrics. The fitted temperature can be written into
the checkpoint config so it travels with the model and is applied at serving time.

Kept dependency-light: torch is imported lazily through the model module, and the
fit itself runs on plain Python lists via :func:`fit_temperature`.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .data import load_examples
from .engine import _prediction_records
from .metrics import _summary, evaluate_workflow_precedence, fit_temperature

def _forward_log_probs(model, tokenizer, examples, *, max_length, device, batch_size):
    """Return raw log-probability rows at temperature 1.0, one per flattened question.

    Using log-probabilities (log of the model's own normalized outputs) is the
    representation the trainer fits on; dividing by a temperature and re-normalising
    is algebraically identical to scaling the raw logits before softmax, so the
    fitted scalar transfers directly to serving.
    """

    records = _prediction_records(
        model, tokenizer, examples, max_length=max_length,
        device=device, batch_size=batch_size, temperature=1.0,
    )
    log_probs = []
    for record in records:
        log_probs.append([math.log(max(1e-12, p)) for p in record["probabilities"]])
    return log_probs, [record["label_index"] for record in records], records


def _softmax_row(logits: Sequence[float], temperature: float) -> list[float]:
    values = [float(v) / temperature for v in logits]
    pivot = max(values)
    exps = [math.exp(v - pivot) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def fit_stability_bootstrap(
    log_probs: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    iterations: int = 100,
    seed: int = 7,
) -> dict[str, float]:
    """Resample the held-out fit and report how stable the temperature is.

    A small held-out split can produce a temperature that is more noise than
    signal; the bootstrap distribution (mean/std/percentiles) quantifies that so
    the summary can say how much trust the fitted value deserves.
    """

    if not log_probs:
        return {"iterations": 0}
    rng = random.Random(seed)
    n = len(log_probs)
    temperatures: list[float] = []
    for _ in range(max(0, iterations)):
        idx = [rng.randrange(n) for _ in range(n)]
        cal = fit_temperature([log_probs[i] for i in idx], [labels[i] for i in idx])
        temperatures.append(cal["temperature"])
    if not temperatures:
        return {"iterations": 0}
    mean = sum(temperatures) / len(temperatures)
    std = math.sqrt(sum((t - mean) ** 2 for t in temperatures) / len(temperatures))
    ordered = sorted(temperatures)

    def pct(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * q
        low = int(math.floor(rank))
        high = min(low + 1, len(ordered) - 1)
        frac = rank - low
        return ordered[low] * (1 - frac) + ordered[high] * frac

    return {
        "iterations": len(temperatures),
        "mean": mean,
        "std": std,
        "min": ordered[0],
        "p10": pct(0.10),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "max": ordered[-1],
    }


def _metrics_at(records, log_probs, temperature):
    """Build before/after-style metric summaries for a given temperature.

    Records are rebuilt from the raw log-probabilities so the same held-out
    questions are scored at temperature 1.0 and at the fitted temperature without
    a second forward pass.
    """

    scaled = []
    for record, lp in zip(records, log_probs):
        scaled.append({
            "kind": record.get("kind", "unknown"),
            "probabilities": _softmax_row(lp, temperature),
            "target_probabilities": record.get("target_probabilities"),
            "label_index": record["label_index"],
        })
    by_kind: dict[str, list[Mapping[str, Any]]] = {}
    for record in scaled:
        by_kind.setdefault(str(record["kind"]), []).append(record)
    return {
        "overall": _summary(scaled),
        "by_kind": {kind: _summary(values) for kind, values in sorted(by_kind.items())},
    }


def calibrate(
    *,
    checkpoint: str | Path,
    data: Iterable[str | Path] | str | Path,
    device: str | None = None,
    batch_size: int = 32,
    max_length: int = 512,
    bootstrap_iterations: int = 100,
    seed: int = 7,
    apply: bool = True,
) -> dict[str, Any]:
    """Fit a held-out temperature and (optionally) persist it into the checkpoint.

    ``data`` may be a single JSONL path or a list of paths (e.g. the two public
    test splits). The fitted temperature, its bootstrap stability, and the
    before/after metrics are returned; when ``apply`` is true the temperature is
    merged into the checkpoint config so it travels with the model.
    """

    from ..distilbert_model import load_checkpoint, save_calibration

    paths = [data] if isinstance(data, (str, Path)) else list(data)
    examples: list = []
    for path in paths:
        examples.extend(load_examples(path))
    if not examples:
        raise ValueError("no held-out examples supplied for calibration")

    model, tokenizer, config = load_checkpoint(checkpoint, device=device)
    start = time.perf_counter()
    log_probs, labels, records = _forward_log_probs(
        model, tokenizer, examples, max_length=max_length,
        device=device, batch_size=batch_size,
    )
    forward_seconds = time.perf_counter() - start

    calibration = fit_temperature(log_probs, labels)
    temperature = calibration["temperature"]
    bootstrap = fit_stability_bootstrap(log_probs, labels, iterations=bootstrap_iterations, seed=seed)

    before = _metrics_at(records, log_probs, 1.0)
    after = _metrics_at(records, log_probs, temperature)
    precedence = evaluate_workflow_precedence()

    report: dict[str, Any] = {
        "format": "jev-laya-calibration-v1",
        "checkpoint": str(checkpoint),
        "split": [str(p) for p in paths],
        "counts": {"examples": len(examples), "questions": len(log_probs)},
        "calibration": calibration,
        "bootstrap": bootstrap,
        "before": before,
        "after": after,
        "deterministic_precedence": precedence,
        "forward_seconds": forward_seconds,
        "applied": False,
    }

    if apply:
        save_calibration(checkpoint, {
            "temperature": temperature,
            "nll_before": calibration["nll_before"],
            "nll_after": calibration["nll_after"],
            "split": [str(p) for p in paths],
            "questions": len(log_probs),
        })
        report["applied"] = True

    return report


__all__ = ["calibrate", "fit_stability_bootstrap"]
