"""Lazy local DistilBERT runtime for the opt-in HTTP backend."""

from __future__ import annotations

import math
import os
from pathlib import Path

from . import schema
from .distilbert_model import DEFAULT_MAX_LENGTH, load_checkpoint, render_question_prompt


def _confidence(probabilities):
    values = list(probabilities.values())
    if len(values) <= 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in values if p > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(values))))


class LocalDistilBertRuntime:
    """Run the trained local checkpoint without Hub access or tool execution."""

    def __init__(self, model_path=None, device=None):
        path_value = model_path or os.environ.get("JEV_DISTILBERT_MODEL_PATH")
        schema.require(path_value, "JEV_DISTILBERT_MODEL_PATH must identify a local checkpoint")
        path = Path(path_value).expanduser()
        schema.require(path.is_dir(), "local-distilbert checkpoint directory does not exist")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.model, self.tokenizer, self.config = load_checkpoint(
            path, device=device or os.environ.get("JEV_DISTILBERT_DEVICE"), local_files_only=True
        )
        self.device = next(self.model.parameters()).device
        self.max_length = int(self.config.get("max_length", DEFAULT_MAX_LENGTH))
        self.max_choice_options = int(self.config.get("max_choice_options", 8))
        self.max_score_options = int(self.config.get("max_score_options", 10))

    def predict(self, state, questions):
        torch = self._torch()
        prompts = []
        names = []
        for name, question in questions.items():
            kind = question["type"]
            count = len(question.get("criteria", {})) if kind == "choice" else len(question.get("criteria", []))
            if kind == "noul":
                count = 2
            limit = self.max_choice_options if kind == "choice" else self.max_score_options
            schema.require(count <= limit, f"local-distilbert supports at most {limit} {kind} labels")
            prompts.append(render_question_prompt(state, question))
            names.append(name)
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded)
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        output = {}
        for index, (name, question) in enumerate(questions.items()):
            kind = question["type"]
            if kind == "choice":
                labels = list(question["criteria"])
                values = probabilities[index][:len(labels)]
                total = sum(values) or 1.0
                values = [value / total for value in values]
                distribution = dict(zip(labels, values))
                output[name] = {
                    "type": kind,
                    "probabilities": distribution,
                    "confidence": _confidence(distribution),
                    "choice": max(distribution, key=distribution.get),
                }
            elif kind == "score":
                labels = [str(i) for i in range(len(question["criteria"]))]
                values = probabilities[index][:len(labels)]
                total = sum(values) or 1.0
                values = [value / total for value in values]
                distribution = dict(zip(labels, values))
                output[name] = {
                    "type": kind,
                    "probabilities": distribution,
                    "confidence": _confidence(distribution),
                    "score": sum(int(label) * value for label, value in distribution.items()),
                    "legend": {str(i): schema.text(value) for i, value in enumerate(question["criteria"])},
                }
            else:
                values = probabilities[index][:2]
                total = sum(values) or 1.0
                output[name] = {"type": kind, "noul": float(values[1] / total)}
        return {
            "answers": output,
            "usage": {
                "input_tokens": int(encoded["attention_mask"].sum().item()),
                "output_tokens": 0,
            },
        }

    @staticmethod
    def _torch():
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("local-distilbert requires PyTorch") from exc
        return torch


__all__ = ["LocalDistilBertRuntime"]
