"""PyTorch training and smoke-test routines for the compact decision head."""

from __future__ import annotations

import contextlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..distilbert_model import (
    DEFAULT_BASE_MODEL,
    DEFAULT_HEAD_DIM,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_OPTIONS,
    build_model,
    load_base_components,
    render_question_prompt,
    save_checkpoint,
)
from .data import DecisionExample, make_synthetic_cases, split_examples
from .metrics import evaluate_predictions, evaluate_workflow_precedence, fit_temperature


def _imports():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("training requires PyTorch; install jev-laya-free[training]") from exc
    return torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        torch = _imports()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except RuntimeError:
        pass


def flatten_examples(examples: Iterable[DecisionExample]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example in examples:
        for name, question in example.questions.items():
            target = example.targets[name]
            records.append({
                "example_id": example.example_id,
                "group": example.group,
                "state": example.state,
                "question_name": name,
                "question": question,
                "target": target,
                "metadata": example.metadata,
            })
    return records


def _batches(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _encode_batch(tokenizer, batch: list[Mapping[str, Any]], *, max_length: int, device):
    prompts = [render_question_prompt(item["state"], item["question"]) for item in batch]
    encoded = tokenizer(
        prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    return {key: value.to(device) for key, value in encoded.items()}


def typed_loss(logits, batch: list[Mapping[str, Any]], *, brier_weight: float = 0.2,
               rps_weight: float = 0.1):
    """Soft-target CE plus Brier and ordinal RPS terms."""

    torch = _imports()
    losses = []
    ce_values = []
    brier_values = []
    rps_values = []
    for row_index, item in enumerate(batch):
        target = item["target"]
        target_values = torch.tensor(target["probabilities"], dtype=logits.dtype, device=logits.device)
        count = int(target_values.numel())
        row_logits = logits[row_index, :count]
        log_probabilities = torch.log_softmax(row_logits, dim=-1)
        probabilities = log_probabilities.exp()
        ce = -(target_values * log_probabilities).sum()
        brier = ((probabilities - target_values) ** 2).sum()
        loss = ce + brier_weight * brier
        ce_values.append(ce.detach())
        brier_values.append(brier.detach())
        if item["question"]["type"] == "score" and count > 1:
            pred_cdf = torch.cumsum(probabilities, dim=0)[:-1]
            target_cdf = torch.cumsum(target_values, dim=0)[:-1]
            rps = ((pred_cdf - target_cdf) ** 2).mean()
            loss = loss + rps_weight * rps
            rps_values.append(rps.detach())
        losses.append(loss)
    total = torch.stack(losses).mean()
    return total, {
        "loss": float(total.detach().cpu()),
        "ce": float(torch.stack(ce_values).mean().cpu()),
        "brier": float(torch.stack(brier_values).mean().cpu()),
        "rps": float(torch.stack(rps_values).mean().cpu()) if rps_values else None,
    }


def _amp_context(torch, device, precision: str):
    if precision not in {"bf16", "fp16"} or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _prediction_records(model, tokenizer, examples: list[DecisionExample], *, max_length: int,
                        device, batch_size: int, temperature: float = 1.0):
    torch = _imports()
    model.eval()
    records = []
    for batch in _batches(flatten_examples(examples), batch_size):
        encoded = _encode_batch(tokenizer, batch, max_length=max_length, device=device)
        with torch.inference_mode():
            logits = model(**encoded).detach().cpu().tolist()
        for row, values in zip(batch, logits):
            target = row["target"]
            count = len(target["probabilities"])
            probabilities = torch.softmax(
                torch.tensor(values[:count], dtype=torch.float32) / temperature, dim=-1
            ).tolist()
            predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
            question_kind = row["question"]["type"]
            record = {
                "example_id": row["example_id"],
                "question_name": row["question_name"],
                "kind": question_kind,
                "probabilities": probabilities,
                "target_probabilities": list(target["probabilities"]),
                "label_index": int(target["label_index"]),
                "predicted_index": predicted,
                "metadata": row.get("metadata", {}),
            }
            if question_kind == "noul" and row.get("metadata", {}).get("repeat_without_progress") is not None:
                record["expected_repeat"] = bool(row["metadata"].get("repeat_without_progress"))
                record["predicted_repeat"] = predicted == 1
            records.append(record)
    return records


def _device(torch, requested: str | None):
    target = ("cuda" if torch.cuda.is_available() else "cpu") if requested in (None, "", "auto") else requested
    device = torch.device(target)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def one_batch_smoke(*, model_name: str = DEFAULT_BASE_MODEL, device: str | None = None,
                    max_length: int = DEFAULT_MAX_LENGTH, head_dim: int = DEFAULT_HEAD_DIM,
                    max_options: int = DEFAULT_MAX_OPTIONS, seed: int = 7,
                    local_files_only: bool = False) -> dict[str, Any]:
    """Load the public base checkpoint and perform one forward/backward optimizer step."""

    torch = _imports()
    seed_everything(seed)
    target_device = _device(torch, device)
    encoder, tokenizer = load_base_components(model_name, local_files_only=local_files_only)
    model = build_model(encoder, head_dim=head_dim, max_options=max_options).to(target_device)
    model.train()
    examples = make_synthetic_cases(4)
    batch = flatten_examples(examples)[:6]
    encoded = _encode_batch(tokenizer, batch, max_length=max_length, device=target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    optimizer.zero_grad(set_to_none=True)
    with _amp_context(torch, target_device, "bf16"):
        logits = model(**encoded)
        loss, components = typed_loss(logits, batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "status": "ok",
        "device": str(target_device),
        "model_name": model_name,
        "batch_questions": len(batch),
        "loss": components,
        "cuda": bool(torch.cuda.is_available()),
    }


def tiny_random_smoke(*, device: str | None = None, seed: int = 7,
                      head_dim: int = DEFAULT_HEAD_DIM, max_options: int = DEFAULT_MAX_OPTIONS) -> dict[str, Any]:
    """Exercise the full tensor/loss path without downloading a checkpoint.

    This is useful for CI and disconnected hosts.  It validates integration only; it is
    not an accuracy or throughput benchmark for the real 67M-parameter encoder.
    """

    torch = _imports()
    try:
        from transformers import DistilBertConfig, DistilBertModel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("tiny smoke requires transformers; install jev-laya-free[training]") from exc
    seed_everything(seed)
    target_device = _device(torch, device)
    config = DistilBertConfig(
        vocab_size=30522, dim=64, hidden_dim=128, n_layers=1, n_heads=2,
        dropout=0.0, attention_dropout=0.0,
    )
    model = build_model(
        DistilBertModel(config), head_dim=head_dim, max_options=max_options
    ).to(target_device)
    model.train()
    examples = flatten_examples(make_synthetic_cases(2))[:4]
    input_ids = torch.randint(0, config.vocab_size, (len(examples), 24), device=target_device)
    attention_mask = torch.ones_like(input_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    loss, components = typed_loss(logits, examples)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "status": "ok", "synthetic": True, "device": str(target_device),
        "cuda": bool(torch.cuda.is_available()), "batch_questions": len(examples),
        "loss": components,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def train(*, examples: list[DecisionExample], output_dir: str | Path,
          validation_examples: list[DecisionExample] | None = None,
          model_name: str = DEFAULT_BASE_MODEL, device: str | None = None,
          epochs: int = 3, warmup_epochs: int = 1, batch_size: int = 8,
          learning_rate: float = 2e-5, max_length: int = DEFAULT_MAX_LENGTH,
          head_dim: int = DEFAULT_HEAD_DIM, max_options: int = DEFAULT_MAX_OPTIONS,
          validation_fraction: float = 0.2, seed: int = 7, precision: str = "none",
          local_files_only: bool = False) -> dict[str, Any]:
    torch = _imports()
    seed_everything(seed)
    if validation_examples is None:
        train_examples, validation_examples = split_examples(examples, validation_fraction)
    else:
        train_examples = list(examples)
        validation_examples = list(validation_examples)
    target_device = _device(torch, device)
    encoder, tokenizer = load_base_components(model_name, local_files_only=local_files_only)
    model = build_model(encoder, head_dim=head_dim, max_options=max_options).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    train_records = flatten_examples(train_examples)
    history = []
    total_epochs = max(1, max(0, warmup_epochs) + max(0, epochs))
    total_steps = max(1, total_epochs * ((len(train_records) + batch_size - 1) // batch_size))
    warmup_steps = max(1, int(total_steps * 0.1))
    step = 0
    for epoch in range(total_epochs):
        freeze_encoder = epoch < warmup_epochs
        for parameter in model.encoder.parameters():
            parameter.requires_grad = not freeze_encoder
        model.train()
        for batch in _batches(train_records, batch_size):
            encoded = _encode_batch(tokenizer, batch, max_length=max_length, device=target_device)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(torch, target_device, precision):
                logits = model(**encoded)
                loss, components = typed_loss(logits, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step <= warmup_steps:
                scale = step / warmup_steps
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate * scale
        validation_records = _prediction_records(
            model, tokenizer, validation_examples, max_length=max_length,
            device=target_device, batch_size=batch_size,
        )
        history.append({"epoch": epoch + 1, "last_loss": components, "validation": evaluate_predictions(validation_records)})
    validation_records = _prediction_records(
        model, tokenizer, validation_examples, max_length=max_length,
        device=target_device, batch_size=batch_size,
    )
    logits_for_temperature = []
    labels_for_temperature = []
    # Fit a calibration scalar from the already-normalized probabilities; log is sufficient
    # for this post-hoc report and keeps no validation tensors in the checkpoint.
    for record in validation_records:
        probabilities = record["probabilities"]
        logits_for_temperature.append([math_log(value) for value in probabilities])
        labels_for_temperature.append(record["label_index"])
    calibration = fit_temperature(logits_for_temperature, labels_for_temperature)
    output_dir = Path(output_dir)
    save_checkpoint(
        model, tokenizer, output_dir, base_model=model_name,
        max_length=max_length, head_dim=head_dim, max_options=max_options,
        metadata={"training_examples": len(train_examples), "validation_examples": len(validation_examples)},
    )
    checkpoint_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    report = {
        "format": "jev-laya-training-report-v1",
        "model": model_name,
        "device": str(target_device),
        "counts": {"examples": len(examples), "train": len(train_examples), "validation": len(validation_examples)},
        "history": history,
        "validation": evaluate_predictions(validation_records),
        "calibration": calibration,
        "deterministic_precedence": evaluate_workflow_precedence(),
        "checkpoint": {"path": str(output_dir), "bytes": checkpoint_bytes,
                       "megabytes": checkpoint_bytes / (1024 * 1024), "storage_dtype": "float16",
                       "parameters": sum(parameter.numel() for parameter in model.parameters())},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def math_log(value: float) -> float:
    import math
    return math.log(max(1e-12, float(value)))


__all__ = ["flatten_examples", "one_batch_smoke", "tiny_random_smoke", "train", "typed_loss"]
