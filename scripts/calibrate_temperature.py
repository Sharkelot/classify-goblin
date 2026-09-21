"""Held-out temperature calibration + advisory re-evaluation for the local DistilBERT checkpoint.

Loads ``checkpoints/local-distilbert``, scores the offline public test sets
(``data/public-test-a.jsonl`` + ``data/public-test-b.jsonl`` — the documented held-out
split, never seen in training), and fits a scalar temperature with the existing
``fit_temperature()`` from the dependency-light metrics module (a single forward pass
supplies the raw logits; ``fit_temperature`` does the grid search over them).

The fitted temperature is written back to the checkpoint config via
``save_calibration`` so it travels with the model and is applied by the serving
backend.  ``reports/calibration.json`` records the fitted temperature, the split used,
the before/after metrics (overall + per-kind), and the deterministic-guard precedence.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
__import__("sys").path.insert(0, str(ROOT / "src"))

from jev_laya_free.distilbert_model import (  # noqa: E402
    calibration_temperature,
    load_checkpoint,
    render_question_prompt,
    save_calibration,
)
from jev_laya_free.training.data import load_examples  # noqa: E402
from jev_laya_free.training.metrics import (  # noqa: E402
    compare_reports,
    evaluate_predictions,
    evaluate_workflow_precedence,
    fit_temperature,
)

CHECKPOINT = ROOT / "checkpoints" / "local-distilbert"
TEST_FILES = [
    ROOT / "data" / "public-test-a.jsonl",
    ROOT / "data" / "public-test-b.jsonl",
]
REPORT = ROOT / "reports" / "calibration.json"
BATCH = 16
BOOTSTRAP = 100


def _records(examples):
    """Yield (example, name, question, target, labels) per question."""
    for example in examples:
        for name, question in example.questions.items():
            target = example.targets[name]
            kind = question["type"]
            if kind == "choice":
                labels = list(question["criteria"])
            elif kind == "score":
                labels = [str(i) for i in range(len(question["criteria"]))]
            else:
                labels = ["false", "true"]
            yield example, name, question, target, labels


def _encode(tokenizer, prompts):
    return tokenizer(
        prompts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )


def _forward_logits(model, tokenizer, records, *, device):
    """Single forward pass -> raw (temperature 1.0) logits per record, truncated to label count."""
    logits_rows = []
    prompts = [
        render_question_prompt(example.state, question)
        for example, _name, question, _target, _labels in records
    ]
    for start in range(0, len(prompts), BATCH):
        batch = prompts[start:start + BATCH]
        encoded = _encode(tokenizer, batch)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            raw = model(**encoded).float()
        for offset, (_example, _name, _question, _target, labels) in enumerate(
            records[start:start + BATCH]
        ):
            logits_rows.append(raw[offset].cpu().tolist()[: len(labels)])
    return logits_rows


def _probs_from_logits(logits, temperature):
    """Stable log-softmax of logits/temperature -> probabilities."""
    scaled = [value / temperature for value in logits]
    pivot = max(scaled)
    exps = [torch.exp(torch.tensor(v - pivot)).item() for v in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def _build_eval_records(records, probs_rows):
    """Model-prediction records for evaluate_predictions (gold label_index, model probabilities)."""
    out = []
    for (_example, name, question, target, labels), probs in zip(records, probs_rows):
        out.append(
            {
                "question_name": name,
                "kind": question["type"],
                "labels": labels,
                "label_index": int(target["label_index"]),
                "probabilities": probs,
                "question": {"criteria": dict(question.get("criteria", {})) if question["type"] == "choice" else list(question.get("criteria", []))},
            }
        )
    return out


def _gold_nll(gold, predicted):
    """Distributional NLL of the model distribution against the gold probability vector."""
    floor = 1e-9
    return -sum(g * math.log(max(p, floor)) for g, p in zip(gold, predicted))


def _bootstrap_stability(logits_rows, label_indices, n=BOOTSTRAP):
    """Resample the held-out set to gauge how stable the fitted temperature is.

    400 rows is a small calibration set, so we report the spread of the fitted
    temperature across bootstrap resamples rather than silently widening the grid.
    """
    import random

    rng = random.Random(20260920)
    n_total = len(logits_rows)
    best_temps = []
    for _ in range(n):
        idx = [rng.randrange(n_total) for _ in range(n_total)]
        sub_logits = [logits_rows[i] for i in idx]
        sub_labels = [label_indices[i] for i in idx]
        fit = fit_temperature(sub_logits, sub_labels)
        best_temps.append(fit["temperature"])
    counts = Counter(round(t, 3) for t in best_temps)
    mode_temp, mode_count = counts.most_common(1)[0]
    return {
        "resamples": n,
        "mode_temperature": mode_temp,
        "mode_support": mode_count,
        "min_temperature": round(min(best_temps), 4),
        "max_temperature": round(max(best_temps), 4),
        "spread": round(max(best_temps) - min(best_temps), 4),
        "stable": mode_count / n >= 0.5,
    }


def main() -> int:
    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_checkpoint(CHECKPOINT, device=device)
    examples = []
    for path in TEST_FILES:
        examples.extend(load_examples(path))
    records = list(_records(examples))
    print(f"loaded {len(examples)} held-out examples -> {len(records)} question records", flush=True)

    # Baseline temperature: read from the calibration block (the key the serving
    # backend actually applies).  Default 1.0 when the block is absent (pre-calibration).
    baseline_temperature = calibration_temperature(CHECKPOINT)
    print(f"baseline temperature={baseline_temperature}", flush=True)

    # Single forward pass -> raw logits.  fit_temperature() does the grid search.
    logits_rows = _forward_logits(model, tokenizer, records, device=device)
    label_indices = [int(target["label_index"]) for _example, _name, _question, target, _labels in records]
    fit = fit_temperature(logits_rows, label_indices)
    fitted_temperature = float(fit["temperature"])
    print(
        f"fit_temperature -> temperature={fitted_temperature} "
        f"nll_before={fit['nll_before']:.6f} nll_after={fit['nll_after']:.6f}",
        flush=True,
    )

    # Before/after model-based metrics (overall + per-kind) on the held-out split.
    base_probs = [_probs_from_logits(row, baseline_temperature) for row in logits_rows]
    cal_probs = [_probs_from_logits(row, fitted_temperature) for row in logits_rows]
    base_report = evaluate_predictions(_build_eval_records(records, base_probs))
    cal_report = evaluate_predictions(_build_eval_records(records, cal_probs))
    comparison = compare_reports(cal_report, base_report)

    # Gold-distribution NLL before/after (advisory; the gold vector is the target).
    gold_vectors = [target["probabilities"] for _example, _name, _question, target, _labels in records]
    base_gold_nll = sum(_gold_nll(g, p) for g, p in zip(gold_vectors, base_probs)) / len(records)
    cal_gold_nll = sum(_gold_nll(g, p) for g, p in zip(gold_vectors, cal_probs)) / len(records)

    # Deterministic guard precedence must remain authoritative and 13/13.
    precedence = evaluate_workflow_precedence()
    print(
        f"deterministic guard precedence {precedence['passed']}/{precedence['total']} "
        f"(evidence routing {precedence['evidence_routing']['passed']}/{precedence['evidence_routing']['total']})",
        flush=True,
    )

    # Bootstrap stability: 400 rows is a small calibration set, so report the spread.
    bootstrap = _bootstrap_stability(logits_rows, label_indices)
    print(f"bootstrap stability: mode={bootstrap['mode_temperature']} "
          f"spread={bootstrap['spread']} stable={bootstrap['stable']}", flush=True)

    # Persist the fitted temperature into the checkpoint config so it travels with the model.
    save_calibration(
        CHECKPOINT,
        {
            "temperature": fitted_temperature,
            "nll_before": fit["nll_before"],
            "nll_after": fit["nll_after"],
            "fitted_on": [str(p.name) for p in TEST_FILES],
        },
    )
    print(f"saved calibration temperature={fitted_temperature} to {CHECKPOINT / 'jev_laya_config.json'}", flush=True)

    report = {
        "advisory_only": True,
        "authorizes_execution": False,
        "checkpoint": str(CHECKPOINT),
        "device": device,
        "held_out_examples": len(examples),
        "held_out_question_records": len(records),
        "test_files": [str(p) for p in TEST_FILES],
        "method": "fit_temperature (grid 0.25-4.0, 61 steps) over single forward-pass logits",
        "baseline_temperature": baseline_temperature,
        "fitted_temperature": fitted_temperature,
        "fit": fit,
        "gold_nll_before": base_gold_nll,
        "gold_nll_after": cal_gold_nll,
        "before": base_report,
        "after": cal_report,
        "comparison": comparison,
        "deterministic_guard_precedence": {
            "passed": precedence["passed"],
            "total": precedence["total"],
            "accuracy": precedence["accuracy"],
            "evidence_routing": {
                "passed": precedence["evidence_routing"]["passed"],
                "total": precedence["evidence_routing"]["total"],
                "accuracy": precedence["evidence_routing"]["accuracy"],
            },
        },
        "bootstrap_stability": bootstrap,
        "seconds": round(time.time() - started, 1),
        "note": "Held-out calibration is advisory; the deterministic guard remains authoritative "
               "(calibration changes only advisory probabilities, not gate/route behavior).",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
