"""Dataset preparation for public typed decisions and local redacted traces."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


DEFAULT_LOCAL_TRACES = Path("/home/coreys/models/laya-sidecar/data/hermes-traces.jsonl")
SECRET_KEY = re.compile(r"(token|secret|password|api[_-]?key|authorization|cookie|private[_-]?key)", re.I)


@dataclass
class DecisionExample:
    """One state/question case with a probability target for every question."""

    example_id: str
    group: str
    source: str
    state: Any
    questions: dict[str, dict[str, Any]]
    targets: dict[str, dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecisionExample":
        required = {"example_id", "group", "source", "state", "questions", "targets"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"example is missing fields: {sorted(missing)}")
        return cls(
            example_id=str(value["example_id"]),
            group=str(value["group"]),
            source=str(value["source"]),
            state=value["state"],
            questions=dict(value["questions"]),
            targets=dict(value["targets"]),
            metadata=dict(value.get("metadata", {})),
        )


def _json_or_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def _safe_local(value: Any, *, key: str = "", depth: int = 0) -> Any:
    """Remove obvious credentials and bound local trace text before it enters a dataset."""

    if depth > 8:
        return "[depth omitted]"
    if SECRET_KEY.search(key):
        return "[redacted]"
    if isinstance(value, dict):
        return {
            str(k): _safe_local(v, key=str(k), depth=depth + 1)
            for k, v in value.items()
            if not SECRET_KEY.search(str(k))
        }
    if isinstance(value, list):
        return [_safe_local(v, depth=depth + 1) for v in value[:128]]
    if isinstance(value, str):
        value = re.sub(r"(?i)(bearer\s+)[^\s]+", r"\1[redacted]", value)
        value = re.sub(r"(?i)(sk|ghp|hf)_[A-Za-z0-9_-]{12,}", r"\1_[redacted]", value)
        return value[:12000]
    return value


def _question(value: Mapping[str, Any]) -> dict[str, Any]:
    kind = value.get("type")
    if kind not in {"choice", "score", "noul"}:
        raise ValueError(f"unsupported question type: {kind!r}")
    criteria = value.get("criteria", value.get("options"))
    if kind == "choice":
        if isinstance(criteria, list):
            criteria = {str(label): None for label in criteria}
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("choice question needs a criteria map or option list")
        criteria = {str(k): _safe_local(v) for k, v in criteria.items()}
    elif kind == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score question needs an ordered criteria list")
        criteria = [_safe_local(v) for v in criteria]
    elif criteria is not None:
        criteria = _safe_local(criteria)
    return {
        "type": kind,
        "instructions": _safe_local(value.get("instructions", "")),
        **({"criteria": criteria} if criteria is not None else {}),
    }


def _distribution(values: Iterable[float], count: int) -> list[float]:
    numbers = [max(0.0, float(value)) for value in values]
    numbers = (numbers + [0.0] * count)[:count]
    total = sum(numbers)
    if total <= 0:
        return [1.0 / count] * count
    return [value / total for value in numbers]


def _index(label: Any, labels: list[str], default: int = 0) -> int:
    if isinstance(label, bool):
        label = str(label).lower()
    if isinstance(label, int) and not isinstance(label, bool):
        return max(0, min(len(labels) - 1, label))
    text = str(label)
    if text in labels:
        return labels.index(text)
    try:
        return max(0, min(len(labels) - 1, int(text)))
    except (TypeError, ValueError):
        return default


def _target(question: Mapping[str, Any], gold: Any) -> dict[str, Any]:
    kind = question["type"]
    criteria = question.get("criteria", {})
    gold = _json_or_value(gold)
    if kind == "choice":
        labels = list(criteria)
    elif kind == "score":
        labels = [str(i) for i in range(len(criteria))]
    else:
        labels = ["false", "true"]

    if isinstance(gold, dict):
        probabilities = gold.get("probabilities")
        label = gold.get("label", gold.get("choice", gold.get("value")))
        if kind == "noul" and "noul" in gold:
            label = bool(float(gold["noul"]) >= 0.5)
    else:
        probabilities = None
        label = gold
    if kind == "noul" and isinstance(label, str):
        label = label.lower() == "true"
    if probabilities is not None and isinstance(probabilities, dict):
        raw = [probabilities.get(label, 0.0) for label in labels]
        probs = _distribution(raw, len(labels))
    else:
        idx = _index(label, labels, default=1 if kind == "noul" and bool(label) else 0)
        probs = [0.0] * len(labels)
        probs[idx] = 1.0
    label_index = max(range(len(probs)), key=probs.__getitem__)
    target: dict[str, Any] = {"probabilities": probs, "label_index": label_index}
    if kind == "score":
        explicit_score = gold.get("score") if isinstance(gold, dict) else None
        target["score"] = float(explicit_score) if isinstance(explicit_score, (int, float)) else sum(
            index * value for index, value in enumerate(probs)
        )
    return target


def normalize_row(row: Mapping[str, Any], *, source: str, ordinal: int) -> DecisionExample:
    state = _json_or_value(row.get("state", row.get("input", "")))
    questions_raw = _json_or_value(row.get("questions", {}))
    gold_raw = _json_or_value(row.get("gold", row.get("answers", {})))
    if not isinstance(questions_raw, dict):
        raise ValueError("dataset row questions must be an object")
    if not isinstance(gold_raw, dict):
        gold_raw = {}
    questions = {str(name): _question(question) for name, question in questions_raw.items()}
    targets = {
        name: _target(question, gold_raw.get(name)) for name, question in questions.items()
    }
    raw_id = row.get("id", row.get("example_id", row.get("task_id", ordinal)))
    safe_id = str(_safe_local(raw_id, key="id"))
    group = row.get("group", row.get("task_id", raw_id))
    safe_group = str(_safe_local(group, key="group"))
    metadata = {
        "source": source,
        "repeat_without_progress": bool(row.get("repeat_without_progress", False)),
    }
    if isinstance(row.get("metadata"), dict):
        for key in ("task_id", "workflow", "expected_gate", "modality"):
            if key in row["metadata"]:
                metadata[key] = _safe_local(row["metadata"][key])
    if "workflow" in row and isinstance(row["workflow"], dict):
        metadata["workflow"] = _safe_local(row["workflow"])
    return DecisionExample(
        example_id=f"{source}-{safe_id}",
        group=f"{source}-{safe_group}",
        source=source,
        state=_safe_local(state) if source == "local" else state,
        questions=questions,
        targets=targets,
        metadata=metadata,
    )


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
            yield value


def write_jsonl(path: str | Path, examples: Iterable[DecisionExample]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def load_examples(path: str | Path) -> list[DecisionExample]:
    return [DecisionExample.from_dict(row) for row in read_jsonl(path)]


def load_public_rows(dataset_id: str = "LocalLLaMA/typed-decisions", *, config: str = "all",
                     split: str = "train") -> list[dict[str, Any]]:
    """Load Hub data or an `hf download` parquet directory.

    The local-directory path keeps preparation usable in an offline training environment;
    Polars is used when available so the base package still has no parquet dependency.
    """

    local = Path(dataset_id).expanduser()
    if local.exists():
        if local.is_file():
            candidates = [local]
        else:
            candidates = sorted((local / config).glob(f"{split}-*.parquet"))
            if not candidates:
                candidates = sorted(local.glob(f"{split}-*.parquet"))
        if not candidates:
            raise RuntimeError(f"no {split} parquet files found below {local}")
        try:
            import polars as pl
            rows: list[dict[str, Any]] = []
            for candidate in candidates:
                rows.extend(pl.read_parquet(candidate).to_dicts())
            return rows
        except (ImportError, ModuleNotFoundError):
            try:
                import pandas as pd
                rows = []
                for candidate in candidates:
                    rows.extend(pd.read_parquet(candidate).to_dict(orient="records"))
                return rows
            except (ImportError, ModuleNotFoundError):
                # `datasets` already brings a parquet reader in the training extra and
                # avoids requiring a separate pandas/polars dependency for HF downloads.
                try:
                    from datasets import load_dataset
                except ImportError as exc:
                    raise RuntimeError(
                        "reading downloaded parquet requires datasets, polars, or pandas"
                    ) from exc
                dataset = load_dataset("parquet", data_files=[str(candidate) for candidate in candidates], split="train")
                return [dict(row) for row in dataset]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "public dataset loading requires datasets; use `hf download` or install "
            "jev-laya-free[training]"
        ) from exc
    dataset = load_dataset(dataset_id, config, split=split) if config else load_dataset(dataset_id, split=split)
    return [dict(row) for row in dataset]


def load_local_rows(path: str | Path = DEFAULT_LOCAL_TRACES) -> list[dict[str, Any]]:
    path = Path(path).expanduser()
    if not path.exists():
        return []
    return list(read_jsonl(path))


def split_examples(examples: Iterable[DecisionExample], validation_fraction: float = 0.2):
    """Split by stable group hash so adjacent trace rows cannot leak across splits."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    train: list[DecisionExample] = []
    validation: list[DecisionExample] = []
    for example in examples:
        digest = hashlib.sha256(example.group.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        (validation if bucket < validation_fraction else train).append(example)
    if not train and validation:
        train.append(validation.pop())
    if not validation and len(train) > 1:
        validation.append(train.pop())
    return train, validation


def make_synthetic_cases(count: int = 24) -> list[DecisionExample]:
    """Create deterministic fixtures covering loop prevention and evidence routing."""

    modalities = ["code", "pdf", "image"]
    cases: list[DecisionExample] = []
    for index in range(max(1, int(count))):
        mode = modalities[index % len(modalities)]
        phase = index % 4
        same_action = 2 if phase == 1 else 3 if phase == 2 else 0
        terminal = phase == 3
        gate_label = "terminal" if terminal else "blocked" if phase == 2 else "repeat_without_progress" if phase == 1 else "progress"
        hand = {
            "code": "inspect_code",
            "pdf": "render_pdf_page" if phase == 1 else "extract_pdf_text",
            "image": "inspect_image",
        }[mode]
        evidence = 0 if phase == 0 else 1 if phase == 1 else 2
        state = {
            "task_id": f"fixture-{index:04d}",
            "modality": mode,
            "action": f"{hand}:page-{index % 3}",
            "result": "no progress" if phase in (1, 2) else "verified artifact",
            "same_action_streak": same_action,
            "context_compactions": 2 if phase == 2 else 0,
            "same_tool_failures": 3 if phase == 2 else 0,
            "terminal": terminal,
        }
        questions = {
            "loop_state": {
                "type": "choice",
                "instructions": "Classify the current workflow state.",
                "criteria": {
                    "progress": "verified new artifact or result",
                    "repeat_without_progress": "same action repeated without progress",
                    "blocked": "retry budget exhausted",
                    "terminal": "terminal result already recorded",
                },
            },
            "next_hand": {
                "type": "choice",
                "instructions": "Choose the next bounded evidence handoff.",
                "criteria": {
                    "inspect_code": "inspect source or tests",
                    "extract_pdf_text": "extract PDF text",
                    "render_pdf_page": "render a selected PDF page",
                    "inspect_image": "inspect image pixels",
                    "synthesize": "synthesize grounded evidence",
                },
            },
            "evidence_quality": {
                "type": "score",
                "instructions": "Rate the evidence available for a grounded answer.",
                "criteria": ["insufficient", "partial", "sufficient"],
            },
            "needs_review": {
                "type": "noul",
                "instructions": "Does the no-progress state need human review?",
            },
        }
        targets = {
            "loop_state": {"probabilities": [float(label == gate_label) for label in ["progress", "repeat_without_progress", "blocked", "terminal"]], "label_index": ["progress", "repeat_without_progress", "blocked", "terminal"].index(gate_label)},
            "next_hand": {"probabilities": [float(label == hand) for label in ["inspect_code", "extract_pdf_text", "render_pdf_page", "inspect_image", "synthesize"]], "label_index": ["inspect_code", "extract_pdf_text", "render_pdf_page", "inspect_image", "synthesize"].index(hand)},
            "evidence_quality": {"probabilities": [float(i == evidence) for i in range(3)], "label_index": evidence, "score": float(evidence)},
            "needs_review": {"probabilities": [float(phase != 1), float(phase == 1)], "label_index": int(phase == 1)},
        }
        cases.append(DecisionExample(
            example_id=f"synthetic-{index:04d}", group=f"synthetic-{index:04d}", source="synthetic",
            state=state, questions=questions, targets=targets,
            metadata={"expected_gate": gate_label, "repeat_without_progress": phase == 1},
        ))
    return cases


def prepare_examples(*, public_dataset: str | None = None, public_config: str = "all",
                     public_split: str = "train", local_traces: str | Path | None = None,
                     include_synthetic: bool = False, synthetic_count: int = 24) -> list[DecisionExample]:
    examples: list[DecisionExample] = []
    if public_dataset:
        examples.extend(
            normalize_row(row, source="public", ordinal=index)
            for index, row in enumerate(load_public_rows(public_dataset, config=public_config, split=public_split))
        )
    if local_traces:
        examples.extend(
            normalize_row(row, source="local", ordinal=index)
            for index, row in enumerate(load_local_rows(local_traces))
        )
    if include_synthetic or not examples:
        examples.extend(make_synthetic_cases(synthetic_count))
    return examples


__all__ = [
    "DEFAULT_LOCAL_TRACES", "DecisionExample", "load_examples", "load_local_rows", "load_public_rows",
    "make_synthetic_cases", "normalize_row", "prepare_examples", "read_jsonl", "split_examples",
    "write_jsonl",
]
