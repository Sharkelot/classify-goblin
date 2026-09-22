"""Optional DistilBERT decision model for local inference.

This module deliberately imports PyTorch and Transformers inside the functions that need
them.  Importing :mod:`classify_goblin` therefore remains a standard-library operation.
The checkpoint format is local and explicit; it is not a hosted classify-goblin/Laya checkpoint.

Only the inference surface lives here: checkpoint loading, configuration, prompt
reconstruction, and the decision head.  Training, base-model download, and checkpoint
mutation are not part of this release; they run in a separate, non-distributed
environment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


CHECKPOINT_VERSION = "classify-goblin-distilbert-v1"
# Existing local checkpoints remain readable during the product rename. New
# manifests and exported checkpoints must use CHECKPOINT_VERSION.
LEGACY_CHECKPOINT_VERSION = "jev-laya-distilbert-v1"
_CONFIG_FILENAMES = ("classify_goblin_config.json", "jev_laya_config.json")
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
            "local-distilbert requires PyTorch; install torch in a separate environment"
        ) from exc
    return torch, nn


def _transformers_import():
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised in environments without extras
        raise RuntimeError(
            "local-distilbert requires transformers; install transformers in a separate environment"
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


def _config_path(path: str | Path) -> Path:
    root = Path(path)
    for filename in _CONFIG_FILENAMES:
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return root / _CONFIG_FILENAMES[0]


def checkpoint_config(path: str | Path) -> dict[str, Any]:
    config_path = _config_path(path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid local-distilbert checkpoint: {config_path}") from exc
    if config.get("format") not in {CHECKPOINT_VERSION, LEGACY_CHECKPOINT_VERSION}:
        raise RuntimeError("unsupported local-distilbert checkpoint format")
    return config


def calibration_temperature(path: str | Path) -> float:
    """Read the stored calibration temperature, defaulting to 1.0 when absent.

    A missing block means the checkpoint was trained before calibration, so old
    checkpoints keep serving uncalibrated probabilities.
    """

    block = checkpoint_config(path).get("calibration") or {}
    value = block.get("temperature", 1.0)
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        return 1.0
    return value


def _load_torch_state(path: Path):
    torch, _ = _torch_import()
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older supported PyTorch releases
        return torch.load(path, map_location="cpu")


def load_checkpoint(path: str | Path, *, device: str | None = None, local_files_only: bool = True):
    """Load a complete local checkpoint for inference."""

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
    "CHECKPOINT_VERSION", "LEGACY_CHECKPOINT_VERSION", "DEFAULT_BASE_MODEL", "DEFAULT_MAX_LENGTH", "DEFAULT_HEAD_DIM",
    "DEFAULT_MAX_OPTIONS", "compact_text", "render_question_prompt", "build_model",
    "load_checkpoint", "checkpoint_config", "calibration_temperature",
]
