"""Per-capability evaluation on the held-out test split.

Loads the trained capability checkpoint, runs inference on each capability's
held-out test records, computes per-capability metrics, and writes
reports/final-capabilities.json with nonzero support + metrics for all 12
capabilities.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jev_laya_free.distilbert_model import load_checkpoint, render_question_prompt
from jev_laya_free.training.engine import _prediction_records
from jev_laya_free.training.metrics import _summary, evaluate_workflow_precedence
from jev_laya_free.training.data import load_examples


def per_capability_eval(
    checkpoint: str,
    test_path: str,
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 512,
    temperature: float = 1.0,
) -> dict:
    """Run per-capability evaluation on the held-out test split."""
    model, tokenizer, config = load_checkpoint(checkpoint, device=device)

    # Load raw JSONL to preserve the top-level 'capability' field
    records = []
    with open(test_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Group by capability
    by_cap = defaultdict(list)
    for r in records:
        by_cap[r["capability"]].append(r)

    # Load as DecisionExample for the engine
    examples = load_examples(test_path)

    # Run inference at the given temperature
    pred_records = _prediction_records(
        model, tokenizer, examples,
        max_length=max_length, device=device,
        batch_size=batch_size, temperature=temperature,
    )

    # Build capability -> records mapping
    # The engine flattens examples into one record per question.
    # We need to map each pred_record back to its capability.
    # The pred_records are in the same order as the flattened examples.
    # Each example has exactly one question (the capability question),
    # so each pred_record corresponds to one example.

    # Rebuild the mapping: for each example, get its capability from the raw JSONL
    # The examples are loaded in the same order as the raw JSONL records
    cap_list = [r["capability"] for r in records]

    results = {}
    for cap in sorted(by_cap.keys()):
        # Find indices of records for this capability
        indices = [i for i, r in enumerate(records) if r["capability"] == cap]
        cap_preds = [pred_records[i] for i in indices]
        summary = _summary(cap_preds)
        results[cap] = {
            "support": len(cap_preds),
            **summary,
        }

    # Overall
    overall = _summary(pred_records)

    # Deterministic precedence
    precedence = evaluate_workflow_precedence()

    return {
        "per_capability": results,
        "overall": overall,
        "deterministic_precedence": precedence,
        "counts": {
            "examples": len(records),
            "questions": len(pred_records),
            "capabilities": len(by_cap),
        },
        "temperature": temperature,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/capability-distilbert")
    ap.add_argument("--test", default="data/capability-benchmark-full/capability-benchmark-test.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temperature", type=float, default=1.1875)
    args = ap.parse_args()

    result = per_capability_eval(
        args.checkpoint, args.test,
        device=args.device, temperature=args.temperature,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
