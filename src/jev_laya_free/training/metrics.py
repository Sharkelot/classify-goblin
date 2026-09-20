"""Dependency-light calibration and workflow metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping


def _normalise(values: Iterable[float]) -> list[float]:
    values = [max(0.0, float(value)) for value in values]
    total = sum(values)
    return [value / total for value in values] if total else [1.0 / len(values)] * len(values)


def _log_softmax(logits: Iterable[float], temperature: float = 1.0) -> list[float]:
    values = [float(value) / temperature for value in logits]
    pivot = max(values)
    exps = [math.exp(value - pivot) for value in values]
    total = sum(exps)
    return [math.log(value / total) for value in exps]


def _nll(probabilities: list[float], label: int) -> float:
    return -math.log(max(1e-12, probabilities[max(0, min(len(probabilities) - 1, label))]))


def _ece(probabilities: list[float], label: int, bins: int = 10) -> float:
    confidence = max(probabilities)
    prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
    bucket = min(bins - 1, int(confidence * bins))
    # A one-example contribution is useful for the streaming evaluator; the aggregate
    # function below supplies the proper bucket counts.
    return abs(confidence - float(prediction == label)) if bucket >= 0 else 0.0


def fit_temperature(logits: Iterable[Iterable[float]], labels: Iterable[int], *, minimum: float = 0.25,
                    maximum: float = 4.0, steps: int = 61) -> dict[str, float]:
    """Fit a scalar temperature by deterministic grid search over validation logits."""

    logits = [list(row) for row in logits]
    labels = [int(label) for label in labels]
    if not logits or len(logits) != len(labels):
        return {"temperature": 1.0, "nll_before": 0.0, "nll_after": 0.0}

    def loss(temperature):
        return sum(-_log_softmax(row, temperature)[max(0, min(len(row) - 1, label))]
                   for row, label in zip(logits, labels)) / len(labels)

    before = loss(1.0)
    best_temperature = 1.0
    best_loss = before
    for index in range(max(2, steps)):
        temperature = minimum + (maximum - minimum) * index / (max(2, steps) - 1)
        candidate = loss(temperature)
        if candidate < best_loss:
            best_temperature, best_loss = temperature, candidate
    return {"temperature": best_temperature, "nll_before": before, "nll_after": best_loss}


def _summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0, "accuracy": None, "nll": None, "brier": None, "ece": None, "rps": None}
    correct = 0
    nll = 0.0
    brier = 0.0
    rps_values: list[float] = []
    bins: dict[int, list[float]] = defaultdict(list)
    for record in records:
        probs = _normalise(record["probabilities"])
        label = int(record["label_index"])
        prediction = max(range(len(probs)), key=probs.__getitem__)
        correct += int(prediction == label)
        nll += _nll(probs, label)
        brier += sum((probability - float(index == label)) ** 2 for index, probability in enumerate(probs))
        if record.get("kind") == "score" and len(probs) > 1:
            pred_cdf = 0.0
            rps = 0.0
            target_probs = _normalise(record.get("target_probabilities", [float(index == label) for index in range(len(probs))]))
            target_cdf = 0.0
            for index in range(len(probs) - 1):
                pred_cdf += probs[index]
                target_cdf += target_probs[index]
                rps += (pred_cdf - target_cdf) ** 2
            rps_values.append(rps / (len(probs) - 1))
        confidence = max(probs)
        bucket = min(9, int(confidence * 10))
        bins[bucket].append(float(confidence - float(prediction == label)))
    ece = sum(abs(sum(values) / len(values)) * len(values) for values in bins.values()) / len(records)
    return {
        "count": len(records),
        "accuracy": correct / len(records),
        "nll": nll / len(records),
        "brier": brier / len(records),
        "ece": ece,
        "rps": sum(rps_values) / len(rps_values) if rps_values else None,
    }


def evaluate_predictions(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate typed predictions and optional routing annotations.

    Records are intentionally simple JSON mappings so a report can be produced by a
    training run and compared with the existing Laya baseline without importing a model.
    """

    from .acceptance import annotations
    records = [annotations(record) for record in records]
    by_kind: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_kind[str(record.get("kind", "unknown"))].append(record)
    report: dict[str, Any] = {
        "overall": _summary(records),
        "by_kind": {kind: _summary(values) for kind, values in sorted(by_kind.items())},
    }
    repeat = [record for record in records if "expected_repeat" in record]
    if repeat:
        positives = [record for record in repeat if bool(record["expected_repeat"])]
        recalled = [record for record in positives if bool(record.get("predicted_repeat"))]
        report["repeat_without_progress"] = {
            "support": len(positives),
            "recall": len(recalled) / len(positives) if positives else None,
            "predicted_positive": sum(bool(record.get("predicted_repeat")) for record in repeat),
        }
    routes = [record for record in records if "expected_hand" in record]
    if routes:
        report["evidence_routing"] = {
            "support": len(routes),
            "accuracy": sum(record.get("predicted_hand") == record.get("expected_hand") for record in routes) / len(routes),
        }
    return report


def compare_reports(student: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Return explicit deltas without deciding that a model is safe to enable."""

    student_metrics = student.get("overall", student.get("validation", {}).get("overall", {}))
    baseline_metrics = baseline.get("overall", baseline.get("validation", {}).get("overall", {}))
    delta = {}
    for key in ("accuracy", "nll", "brier", "ece", "rps"):
        left, right = student_metrics.get(key), baseline_metrics.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            delta[f"{key}_delta"] = left - right
    student_repeat = student.get("repeat_without_progress", student.get("validation", {}).get("repeat_without_progress", {}))
    baseline_repeat = baseline.get("repeat_without_progress", baseline.get("validation", {}).get("repeat_without_progress", {}))
    if isinstance(student_repeat.get("recall"), (int, float)) and isinstance(baseline_repeat.get("recall"), (int, float)):
        delta["repeat_without_progress_recall_delta"] = student_repeat["recall"] - baseline_repeat["recall"]
    return {
        "student": student_metrics,
        "baseline": baseline_metrics,
        "deltas": delta,
        "note": "This comparison is advisory; deterministic workflow rules remain authoritative.",
    }


def evaluate_workflow_precedence() -> dict[str, Any]:
    """Verify that deterministic guard precedence is intact for representative fixtures."""

    from ..workflow import guard, route, decide

    cases = [
        ({"terminal": True, "same_action_streak": 9}, "terminal"),
        ({"same_action_streak": 3}, "stop"),
        ({"context_compactions": 2}, "stop"),
        ({"same_tool_failures": 3}, "stop"),
        ({"same_action_streak": 2}, "review"),
        ({"same_action_streak": 3, "artifact_delta": True}, "allow"),
    ]
    results = [{"state": state, "expected": expected, "actual": guard(state)["decision"]} for state, expected in cases]
    class AdversarialClient:
        def system_one(self, **kwargs):
            return {"answers": {"next_hand": "stop", "authorize_write": True, "retry": True}}

    for state, expected in cases + [({"modality": "code"}, "allow")]:
        baseline = decide(state)
        actual = decide(state, AdversarialClient())
        results.append({"state": state, "expected": True,
                        "actual": actual["gate"] == baseline["gate"] and actual["route"] == baseline["route"]
                        and actual["advisory"]["authoritative"] is False})
    passed = sum(result["expected"] == result["actual"] for result in results)
    evidence_cases = [
        ({"modality": "code"}, "inspect_code"),
        ({"modality": "code", "test_needed": True}, "run_test"),
        ({"modality": "pdf"}, "extract_pdf_text"),
        ({"modality": "pdf", "extraction_quality": "scanned"}, "render_pdf_page"),
        ({"modality": "image"}, "inspect_image"),
        ({"source_digest": "sha256:fixture", "location": "page 1", "evidence_sufficient": True, "source_grounded": True}, "synthesize"),
    ]
    route_results = [{"state": state, "expected": expected, "actual": route(state, guard(state))["hand"]} for state, expected in evidence_cases]
    route_passed = sum(result["expected"] == result["actual"] for result in route_results)
    return {
        "passed": passed,
        "total": len(results),
        "accuracy": passed / len(results),
        "cases": results,
        "evidence_routing": {"passed": route_passed, "total": len(route_results), "accuracy": route_passed / len(route_results), "cases": route_results},
    }


__all__ = ["compare_reports", "evaluate_predictions", "evaluate_workflow_precedence", "fit_temperature"]
