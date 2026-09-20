"""Optional DistilBERT decision model shared by training and local inference.

This module deliberately imports PyTorch and Transformers inside the functions that need
them.  Importing :mod:`jev_laya_free` therefore remains a standard-library operation.
The checkpoint format is local and explicit; it is not a hosted Jev/Laya checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


CHECKPOINT_VERSION = "jev-laya-distilbert-v1"
DEFAULT_BASE_MODEL = "distilbert/distilbert-base-uncased"
DEFAULT_MAX_LENGTH = 512
DEFAULT_HEAD_DIM = 192
DEFAULT_MAX_OPTIONS = 10


def compact_text(value: Any, limit: int = 12000) -> str:
    """Serialize a state/question value without creating an unbounded prompt."""

    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + " …"


def render_question_prompt(state: Any, question: Mapping[str, Any]) -> str:
    """Render one typed question as a stable text classification prompt."""

    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice":
        criterion_text = " | ".join(
            f"{key}: {compact_text(value, 512)}" for key, value in criteria.items()
        )
    elif kind == "score":
        criterion_text = " | ".join(
            f"{index}: {compact_text(value, 512)}" for index, value in enumerate(criteria)
        )
    else:
        criterion_text = "true/false"
        if criteria:
            criterion_text = " | ".join(
                f"{key}: {compact_text(value, 512)}" for key, value in criteria.items()
            )
    return (
        "STATE:\n"
        + compact_text(state)
        + "\nQUESTION TYPE: "
        + kind
        + "\nINSTRUCTIONS:\n"
        + compact_text(question["instructions"], 4096)
        + "\nCRITERIA:\n"
        + criterion_text
    )


def _torch_import():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - exercised in environments without extras
        raise RuntimeError(
            "local-distilbert requires the optional training dependencies; "
            "install jev-laya-free[training] in a separate environment"
        ) from exc
    return torch, nn


def _transformers_import():
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised in environments without extras
        raise RuntimeError(
            "local-distilbert requires transformers; "
            "install jev-laya-free[training] in a separate environment"
        ) from exc
    return AutoModel, AutoTokenizer


def _mean_pool(last_hidden_state, attention_mask):
    torch, _ = _torch_import()
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    return summed / mask.sum(dim=1).clamp_min(1.0)


def build_model(encoder, *, head_dim: int = DEFAULT_HEAD_DIM,
                max_options: int = DEFAULT_MAX_OPTIONS, dropout: float = 0.1):
    """Build the small typed head around an already loaded encoder.

    ``head_dim`` is the single hidden decision-head layer.  The final projection is
    required to emit up to ten ordinal/class labels and is shared by all question types.
    """

    torch, nn = _torch_import()

    class DistilBertDecisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            hidden_size = int(getattr(encoder.config, "hidden_size", getattr(encoder.config, "dim", 768)))
            self.projection = nn.Linear(hidden_size, head_dim)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.output = nn.Linear(head_dim, max_options)
            self.head_dim = head_dim
            self.max_options = max_options

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = _mean_pool(outputs.last_hidden_state, attention_mask)
            hidden = self.dropout(self.activation(self.projection(pooled)))
            return self.output(hidden)

    return DistilBertDecisionModel()


def load_base_components(model_name: str = DEFAULT_BASE_MODEL, *, local_files_only: bool = False):
    """Load an encoder/tokenizer from the local Transformers cache or Hub."""

    AutoModel, AutoTokenizer = _transformers_import()
    encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    return encoder, tokenizer


def checkpoint_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path) / "jev_laya_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid local-distilbert checkpoint: {config_path}") from exc
    if config.get("format") != CHECKPOINT_VERSION:
        raise RuntimeError("unsupported local-distilbert checkpoint format")
    return config


def save_checkpoint(model, tokenizer, path: str | Path, *, base_model: str,
                    max_length: int = DEFAULT_MAX_LENGTH, head_dim: int = DEFAULT_HEAD_DIM,
                    max_options: int = DEFAULT_MAX_OPTIONS, metadata: Mapping[str, Any] | None = None,
                    weight_dtype: str = "float16"):
    """Save encoder, tokenizer, head weights, and a small auditable config."""

    torch, _ = _torch_import()
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if weight_dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("weight_dtype must be float32, float16, or bfloat16")
    if weight_dtype == "float16":
        model.half()
    elif weight_dtype == "bfloat16":
        model.bfloat16()
    else:
        model.float()
    encoder_path = path / "encoder"
    encoder_path.mkdir(exist_ok=True)
    model.encoder.save_pretrained(encoder_path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    torch.save(
        {"projection": model.projection.state_dict(), "output": model.output.state_dict()},
        path / "head.pt",
    )
    config = {
        "format": CHECKPOINT_VERSION,
        "base_model": base_model,
        "max_length": int(max_length),
        "head_dim": int(head_dim),
        "max_options": int(max_options),
        "max_choice_options": 8,
        "max_score_options": 10,
        "weight_dtype": weight_dtype,
        "metadata": dict(metadata or {}),
    }
    (path / "jev_laya_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_torch_state(path: Path):
    torch, _ = _torch_import()
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older supported PyTorch releases
        return torch.load(path, map_location="cpu")


def load_checkpoint(path: str | Path, *, device: str | None = None, local_files_only: bool = True):
    """Load a complete local checkpoint for inference or continued training."""

    torch, _ = _torch_import()
    AutoModel, AutoTokenizer = _transformers_import()
    path = Path(path)
    config = checkpoint_config(path)
    encoder_dir = path / "encoder"
    encoder = AutoModel.from_pretrained(encoder_dir, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_files_only)
    model = build_model(
        encoder,
        head_dim=int(config["head_dim"]),
        max_options=int(config["max_options"]),
    )
    state = _load_torch_state(path / "head.pt")
    model.projection.load_state_dict(state["projection"])
    model.output.load_state_dict(state["output"])
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if target.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for local-distilbert but is unavailable")
    model.to(target)
    # Half/bfloat16 storage is useful on the target GPU, while CPU inference uses float32
    # because many CPU kernels do not implement all DistilBERT half operations.
    weight_dtype = config.get("weight_dtype", "float32")
    if str(target).startswith("cuda") and weight_dtype == "float16":
        model.half()
    elif str(target).startswith("cuda") and weight_dtype == "bfloat16":
        model.bfloat16()
    else:
        model.float()
    model.eval()
    return model, tokenizer, config


__all__ = [
    "CHECKPOINT_VERSION", "DEFAULT_BASE_MODEL", "DEFAULT_MAX_LENGTH", "DEFAULT_HEAD_DIM",
    "DEFAULT_MAX_OPTIONS", "compact_text", "render_question_prompt", "build_model",
    "load_base_components", "load_checkpoint", "save_checkpoint", "checkpoint_config",
]
