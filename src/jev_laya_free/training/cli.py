"""Command line entry points for data preparation, GPU smoke, training, and reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import (
    DEFAULT_LOCAL_TRACES,
    load_examples,
    make_synthetic_cases,
    prepare_examples,
    split_examples,
    write_jsonl,
)
from .engine import one_batch_smoke, tiny_random_smoke, train
from .calibrate import calibrate
from .acceptance import AcceptanceConfig, evaluate_acceptance
from .metrics import compare_reports, evaluate_predictions, evaluate_workflow_precedence
from ..distilbert_model import calibration_temperature


def _resolve_calibration(value: str | None) -> float:
    """Resolve a --calibration value into a temperature.

    Accepts either a numeric temperature (e.g. "1.25") or a checkpoint path, in
    which case the stored calibration temperature is read back (default 1.0 when
    the checkpoint was never calibrated, so old checkpoints still evaluate).
    """
    if value is None:
        return 1.0
    try:
        return float(value)
    except ValueError:
        return calibration_temperature(value)


def _rescale(records: list[dict], temperature: float) -> list[dict]:
    """Re-scale stored probabilities through a temperature.

    predictions.jsonl holds normalized probabilities, not raw logits; recovering
    the log-probabilities, dividing by the temperature, and re-normalising is
    algebraically identical to scaling the raw logits before softmax, so the
    same fitted scalar applies end-to-end.
    """
    import math
    rescaled = []
    for record in records:
        probs = record.get("probabilities")
        if not probs:
            rescaled.append(record)
            continue
        log_probs = [math.log(max(1e-12, p)) for p in probs]
        scaled = [lp / temperature for lp in log_probs]
        pivot = max(scaled)
        exps = [math.exp(v - pivot) for v in scaled]
        total = sum(exps)
        probabilities = [e / total for e in exps]
        predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
        new_record = dict(record)
        new_record["probabilities"] = probabilities
        new_record["predicted_index"] = predicted
        rescaled.append(new_record)
    return rescaled


def _common_model_args(parser):
    parser.add_argument("--model", default="distilbert/distilbert-base-uncased", help="HF model ID or local cache path")
    parser.add_argument("--device", default=None, help="cpu, cuda, or a specific CUDA device")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=192)
    parser.add_argument("--max-options", type=int, default=10)
    parser.add_argument("--local-files-only", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m jev_laya_free.training")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="normalize public/local/synthetic cases into JSONL")
    prepare.add_argument("--public-dataset", default=None)
    prepare.add_argument("--public-config", default="all")
    prepare.add_argument("--public-split", default="train")
    prepare.add_argument("--local-traces", default=str(DEFAULT_LOCAL_TRACES), help=f"JSONL path (default: {DEFAULT_LOCAL_TRACES} when it exists)")
    prepare.add_argument("--no-local-traces", action="store_true", help="do not read the default/local Hermes JSONL")
    prepare.add_argument("--include-synthetic", action="store_true")
    prepare.add_argument("--synthetic-count", type=int, default=24)
    prepare.add_argument("--output", default="data/hybrid.jsonl")
    prepare.add_argument("--validation-output", default=None)
    prepare.add_argument("--validation-fraction", type=float, default=0.2)

    smoke = commands.add_parser("smoke", help="one forward/backward/optimizer step")
    _common_model_args(smoke)
    smoke.add_argument("--seed", type=int, default=7)
    smoke.add_argument("--tiny-random", action="store_true", help="use a random one-layer encoder; no Hub assets required")

    fitting = commands.add_parser("train", help="train and calibrate a local checkpoint")
    fitting.add_argument("--data", default="data/hybrid.jsonl")
    fitting.add_argument("--validation-data", default=None, help="separate grouped validation JSONL from `prepare --validation-output`")
    fitting.add_argument("--output-dir", default="checkpoints/local-distilbert")
    _common_model_args(fitting)
    fitting.add_argument("--epochs", type=int, default=3)
    fitting.add_argument("--warmup-epochs", type=int, default=1)
    fitting.add_argument("--batch-size", type=int, default=8)
    fitting.add_argument("--learning-rate", type=float, default=2e-5)
    fitting.add_argument("--precision", choices=("none", "bf16", "fp16"), default="none")
    fitting.add_argument("--validation-fraction", type=float, default=0.2)
    fitting.add_argument("--seed", type=int, default=7)

    evaluate = commands.add_parser("evaluate", help="evaluate prediction JSONL and guard precedence")
    evaluate.add_argument("--predictions", default=None, help="JSONL records emitted by a model evaluator")
    evaluate.add_argument("--baseline", default=None, help="JSON report from current Laya for advisory deltas")
    evaluate.add_argument("--output", default=None)
    evaluate.add_argument("--acceptance-config", help="JSON object of AcceptanceConfig thresholds")
    evaluate.add_argument("--require-acceptance", action="store_true", help="exit 1 when a required gate fails")
    evaluate.add_argument("--calibration", default=None,
                        help="temperature (e.g. 1.25) or checkpoint path; re-scales prediction probabilities through it and reports a with/without side-by-side")

    cal = commands.add_parser("calibrate", help="fit a held-out temperature on a trained checkpoint")
    cal.add_argument("--checkpoint", default="checkpoints/local-distilbert")
    cal.add_argument("--data", nargs="+", required=True, help="held-out JSONL path(s), e.g. data/public-test-a.jsonl data/public-test-b.jsonl")
    cal.add_argument("--device", default=None, help="cpu, cuda, or a specific CUDA device")
    cal.add_argument("--batch-size", type=int, default=32)
    cal.add_argument("--max-length", type=int, default=512)
    cal.add_argument("--bootstrap-iterations", type=int, default=100)
    cal.add_argument("--seed", type=int, default=7)
    cal.add_argument("--output", default=None, help="write the calibration report to this JSON path")
    cal.add_argument("--no-apply", action="store_true", help="do not persist the temperature into the checkpoint config")

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            local = None if args.no_local_traces else args.local_traces
            examples = prepare_examples(
                public_dataset=args.public_dataset,
                public_config=args.public_config,
                public_split=args.public_split,
                local_traces=local,
                include_synthetic=args.include_synthetic,
                synthetic_count=args.synthetic_count,
            )
            train_examples, validation_examples = split_examples(examples, args.validation_fraction)
            count = write_jsonl(args.output, train_examples)
            if args.validation_output:
                write_jsonl(args.validation_output, validation_examples)
            print(json.dumps({"output": args.output, "count": count, "validation_count": len(validation_examples), "sources": sorted({e.source for e in examples})}, indent=2))
            return 0
        if args.command == "smoke":
            if args.tiny_random:
                result = tiny_random_smoke(device=args.device, seed=args.seed,
                                           head_dim=args.head_dim, max_options=args.max_options)
            else:
                result = one_batch_smoke(
                    model_name=args.model, device=args.device, max_length=args.max_length,
                    head_dim=args.head_dim, max_options=args.max_options,
                    seed=args.seed, local_files_only=args.local_files_only,
                )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "train":
            examples = load_examples(args.data)
            validation_examples = load_examples(args.validation_data) if args.validation_data else None
            report = train(
                examples=examples, validation_examples=validation_examples,
                output_dir=args.output_dir, model_name=args.model,
                device=args.device, epochs=args.epochs, warmup_epochs=args.warmup_epochs,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                max_length=args.max_length, head_dim=args.head_dim, max_options=args.max_options,
                validation_fraction=args.validation_fraction, seed=args.seed,
                precision=args.precision, local_files_only=args.local_files_only,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if args.command == "evaluate":
            records = []
            if args.predictions:
                records = [json.loads(line) for line in Path(args.predictions).read_text(encoding="utf-8").splitlines() if line.strip()]
            eval_records = [r for r in records if "probabilities" in r and "label_index" in r]
            prediction_report = evaluate_predictions(eval_records)
            config = AcceptanceConfig(**json.loads(Path(args.acceptance_config).read_text())) if args.acceptance_config else AcceptanceConfig()
            report = {"predictions": prediction_report, "deterministic_precedence": evaluate_workflow_precedence()}
            report["acceptance"] = evaluate_acceptance(records, config)
            if args.baseline:
                report["comparison"] = compare_reports(prediction_report, json.loads(Path(args.baseline).read_text(encoding="utf-8")))
            if args.calibration is not None:
                temperature = _resolve_calibration(args.calibration)
                calibrated = _rescale(eval_records, temperature)
                calibrated_report = evaluate_predictions(calibrated)
                report["calibration"] = {
                    "temperature": temperature,
                    "without_temperature": prediction_report,
                    "with_temperature": calibrated_report,
                    "delta": compare_reports(calibrated_report, prediction_report),
                }
            text = json.dumps(report, indent=2, sort_keys=True)
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(text + "\n", encoding="utf-8")
            print(text)
            return 1 if args.require_acceptance and not report["acceptance"]["passed"] else 0
        if args.command == "calibrate":
            report = calibrate(
                checkpoint=args.checkpoint,
                data=args.data,
                device=args.device,
                batch_size=args.batch_size,
                max_length=args.max_length,
                bootstrap_iterations=args.bootstrap_iterations,
                seed=args.seed,
                apply=not args.no_apply,
            )
            text = json.dumps(report, indent=2, sort_keys=True)
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(text + "\n", encoding="utf-8")
            print(text)
            return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


__all__ = ["build_parser", "main"]
