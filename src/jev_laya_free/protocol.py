"""Machine-readable specification for the Jev-compatible typed-decision protocol.

This module is the single source of truth for the wire contract.  The
reference service (``reference_service``) and the client (``client``) both
validate against the constants and helpers defined here.

The protocol is intentionally small: one POST endpoint, bounded JSON bodies,
and a fixed set of question/answer shapes.  No streaming, no chunked
transfer, no non-loopback transport.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

ENDPOINT = "/v1/systemone"
HTTP_METHOD = "POST"

# ---------------------------------------------------------------------------
# Transport rules
# ---------------------------------------------------------------------------

CONTENT_TYPE = "application/json"
ALLOWED_SCHEMES = ("http",)
ALLOWED_HOSTNAMES = ("127.0.0.1", "localhost")
MAX_CONTENT_LENGTH_DIGITS = 10
CHUNKED_TE_HEADER = "Transfer-Encoding"

# ---------------------------------------------------------------------------
# Size limits
# ---------------------------------------------------------------------------

MAX_REQUEST_BYTES = 65_536
MAX_RESPONSE_BYTES = 1_048_576
MAX_NESTING_DEPTH = 16
MAX_QUESTION_COUNT = 32
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
MIN_SCORE_LEVELS = 2
MAX_DESCRIPTION_BYTES = 4096
MAX_STATE_BYTES = 32_768
MAX_MODEL_NAME_LENGTH = 128
MAX_QUESTION_NAME_LENGTH = 128
MAX_OPTION_NAME_LENGTH = 128

# ---------------------------------------------------------------------------
# Question / answer types
# ---------------------------------------------------------------------------

QUESTION_TYPES = ("choice", "score", "noul")

# ---------------------------------------------------------------------------
# Capability families and legacy questions
# ---------------------------------------------------------------------------

CAPABILITY_FAMILIES: List[str] = [
    "model_routing",
    "confidence_action",
    "tool_screening",
    "progress",
    "completion",
    "skill_selection",
    "compaction",
    "citation",
    "rag_filter",
    "semantic_find",
    "composite",
    "intent_routing",
]

LEGACY_QUESTIONS: List[str] = ["next_hand", "needs_review"]

ALL_QUESTION_NAMES: List[str] = CAPABILITY_FAMILIES + LEGACY_QUESTIONS

# ---------------------------------------------------------------------------
# HTTP status codes
# ---------------------------------------------------------------------------

STATUS_OK = 200
STATUS_BAD_REQUEST = 400
STATUS_UNAUTHORIZED = 401
STATUS_NOT_FOUND = 404
STATUS_UNSUPPORTED_MEDIA = 415
STATUS_UNPROCESSABLE = 422
STATUS_REQUEST_TOO_LARGE = 413
STATUS_REQUEST_TIMEOUT = 408
STATUS_BACKEND_BUSY = 529
STATUS_BAD_GATEWAY = 502
STATUS_SERVICE_UNAVAILABLE = 503

RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504, 529})

# ---------------------------------------------------------------------------
# Guard decisions (deterministic workflow)
# ---------------------------------------------------------------------------

GUARD_DECISIONS = ("allow", "stop", "review", "terminal")

# ---------------------------------------------------------------------------
# Workflow state field categories
# ---------------------------------------------------------------------------

STATE_BOOLS = frozenset({
    "terminal", "artifact_delta", "board_delta", "result_changed",
    "same_action", "evidence_sufficient", "source_grounded", "test_needed",
})
STATE_COUNTS = frozenset({
    "same_action_streak", "same_tool_failures", "context_compactions",
})
STATE_TEXTS = frozenset({
    "status", "modality", "source_digest", "location",
    "extraction_quality", "artifact_digest", "previous_artifact_digest",
    "board_digest", "previous_board_digest",
})
STATE_VALUES = frozenset({
    "action", "previous_action", "result", "previous_result",
})

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _is_json_scalar(value: Any) -> bool:
    return (
        value is None
        or isinstance(value, (str, int, float, bool))
    )


def validate_json_value(value: Any, depth: int = 0) -> None:
    """Recursively validate that *value* is a bounded JSON value.

    Raises ``ValueError`` if any constraint is violated.
    """
    if depth > MAX_NESTING_DEPTH:
        raise ValueError(f"JSON nesting exceeds {MAX_NESTING_DEPTH} levels")
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError("JSON keys must be strings")
            validate_json_value(v, depth + 1)
    elif isinstance(value, list):
        for v in value:
            validate_json_value(v, depth + 1)
    else:
        if not _is_json_scalar(value):
            raise ValueError(f"non-JSON value: {type(value).__name__}")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite JSON number")


def validate_probability(value: Any) -> None:
    """Validate that *value* is a probability in [0, 1]."""
    if not (isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1):
        raise ValueError(f"invalid probability: {value!r}")


def validate_question(question: Dict[str, Any]) -> None:
    """Validate a single question object.

    Raises ``ValueError`` if the question does not conform to the protocol.
    """
    if not isinstance(question, dict):
        raise ValueError("question must be an object")
    allowed_keys = {"type", "instructions", "criteria", "options"}
    extra = set(question) - allowed_keys
    if extra:
        raise ValueError(f"unknown question field(s): {extra}")
    qtype = question.get("type")
    if qtype not in QUESTION_TYPES:
        raise ValueError(f"unsupported question type: {qtype!r}")
    if "instructions" not in question:
        raise ValueError("instructions required")
    _validate_description(question["instructions"])

    if qtype == "choice":
        has_options = "options" in question
        has_criteria = "criteria" in question
        if has_options == has_criteria:
            raise ValueError("choice needs exactly one of options/criteria")
        options = question.get("options", question.get("criteria"))
        if isinstance(options, list):
            if not all(isinstance(k, str) for k in options):
                raise ValueError("choice list must contain strings")
            if len(set(options)) != len(options):
                raise ValueError("duplicate option")
            options = dict.fromkeys(options)
        if not isinstance(options, dict):
            raise ValueError("choice options must be a dict or list")
        if not (1 <= len(options) <= MAX_CHOICE_OPTIONS):
            raise ValueError(f"require 1..{MAX_CHOICE_OPTIONS} options")
        for k, v in options.items():
            if not isinstance(k, str) or not (0 < len(k) <= MAX_OPTION_NAME_LENGTH):
                raise ValueError("invalid option name")
            _validate_description(v, nullable=True)

    elif qtype == "score":
        if "options" in question:
            raise ValueError("score does not accept options")
        levels = question.get("criteria")
        if not isinstance(levels, list) or not (MIN_SCORE_LEVELS <= len(levels) <= MAX_SCORE_LEVELS):
            raise ValueError(f"score needs {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels")
        for level in levels:
            _validate_description(level)

    elif qtype == "noul":
        if "options" in question:
            raise ValueError("noul does not accept options")
        if "criteria" in question:
            criteria = question["criteria"]
            if not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}:
                raise ValueError("invalid noul criteria")
            for v in criteria.values():
                _validate_description(v)


def normalize_question(question: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a validated question into the canonical wire form.

    - ``choice``: ``options`` (list or dict) → ``criteria`` (dict)
    - ``score``: unchanged (``criteria`` is a list)
    - ``noul``: unchanged (``criteria`` is an optional dict)

    The input must already be validated by :func:`validate_question`.
    """
    qtype = question["type"]
    if qtype == "choice":
        options = question.get("options", question.get("criteria"))
        if isinstance(options, list):
            options = dict.fromkeys(options)
        return {"type": qtype, "instructions": question["instructions"],
                "criteria": options}
    return {"type": qtype, "instructions": question["instructions"],
            **({"criteria": question["criteria"]} if "criteria" in question else {})}


def normalize_questions(questions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Normalize a mapping of question names to validated questions."""
    return {name: normalize_question(q) for name, q in questions.items()}


def _validate_description(value: Any, nullable: bool = False) -> None:
    if not ((nullable and value is None) or isinstance(value, (str, dict, list))):
        raise ValueError("description must be string, object, or array")
    if not isinstance(value, str):
        import json
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    else:
        encoded = value.encode()
    if len(encoded) > MAX_DESCRIPTION_BYTES:
        raise ValueError(f"description exceeds {MAX_DESCRIPTION_BYTES} bytes")


def validate_request(body: Any) -> None:
    """Validate a full request payload against the protocol spec.

    Raises ``ValueError`` if any constraint is violated.
    """
    if not isinstance(body, dict):
        raise ValueError("request must be an object")
    allowed = {"model", "state", "questions", "artifacts"}
    extra = set(body) - allowed
    if extra:
        raise ValueError(f"unknown request field(s): {extra}")
    missing = {"model", "state", "questions"} - set(body)
    if missing:
        raise ValueError(f"missing required field(s): {missing}")

    # model
    model = body["model"]
    if not isinstance(model, str) or not (0 < len(model) <= MAX_MODEL_NAME_LENGTH):
        raise ValueError("invalid model")

    # state
    state = body["state"]
    if not isinstance(state, (str, dict, list)):
        raise ValueError("state must be string, object, or array")
    validate_json_value(state)

    # questions
    questions = body["questions"]
    if not isinstance(questions, dict) or not (1 <= len(questions) <= MAX_QUESTION_COUNT):
        raise ValueError(f"require 1..{MAX_QUESTION_COUNT} questions")
    for name, q in questions.items():
        if not isinstance(name, str) or not (0 < len(name) <= MAX_QUESTION_NAME_LENGTH):
            raise ValueError("invalid question name")
        validate_question(q)

    # size
    import json
    encoded = json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError(f"request exceeds {MAX_REQUEST_BYTES} bytes")


def validate_answer(answer: Any, question: Dict[str, Any]) -> None:
    """Validate a single answer against its question.

    Raises ``ValueError`` if the answer does not conform.
    """
    if not isinstance(answer, dict):
        raise ValueError("answer must be an object")
    qtype = question["type"]
    if answer.get("type") != qtype:
        raise ValueError("answer type mismatch")

    if qtype == "noul":
        validate_probability(answer.get("noul"))
        return

    validate_probability(answer.get("confidence"))
    if qtype == "choice":
        keys = list(question["criteria"].keys())
    else:
        keys = [str(i) for i in range(len(question["criteria"]))]

    p = answer.get("probabilities")
    if not isinstance(p, dict) or set(p) != set(keys):
        raise ValueError("probability keys mismatch")
    for v in p.values():
        validate_probability(v)
    if abs(sum(p.values()) - 1) > 0.002:
        raise ValueError("probabilities must sum to one")

    if qtype == "choice":
        choice = answer.get("choice")
        if not isinstance(choice, str) or choice not in p:
            raise ValueError("invalid choice")
        if p[choice] != max(p.values()):
            raise ValueError("choice must maximize probability")
    else:
        score = answer.get("score")
        if not (isinstance(score, (int, float)) and math.isfinite(score)
                and 0 <= score <= len(keys) - 1):
            raise ValueError("invalid score")
        expected = sum(int(k) * p[k] for k in keys)
        if abs(score - expected) > 0.02:
            raise ValueError("score must match weighted mean")


def validate_response(body: Any, questions: Dict[str, Any]) -> None:
    """Validate a full response payload against the protocol spec.

    Raises ``ValueError`` if any constraint is violated.
    """
    if not isinstance(body, dict):
        raise ValueError("response must be an object")
    if not (isinstance(body.get("model"), str) and body["model"]):
        raise ValueError("response model required")
    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("usage required")
    for key in ("input_tokens", "output_tokens"):
        if not (isinstance(usage.get(key), int) and usage[key] >= 0):
            raise ValueError("invalid usage")
    if "request_id" in body:
        if not (isinstance(body["request_id"], str) and body["request_id"]):
            raise ValueError("invalid request_id")
    answers = body.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("answer names mismatch")
    for name, q in questions.items():
        validate_answer(answers[name], q)


def validate_transport(
    method: str,
    path: str,
    content_type: Optional[str],
    content_length: Optional[str],
    has_chunked: bool,
) -> None:
    """Validate transport-level constraints.

    Raises ``ValueError`` if any constraint is violated.
    """
    if method != HTTP_METHOD:
        raise ValueError(f"method must be {HTTP_METHOD}")
    if path != ENDPOINT:
        raise ValueError(f"path must be {ENDPOINT}")
    if content_type != CONTENT_TYPE:
        raise ValueError(f"Content-Type must be {CONTENT_TYPE}")
    if has_chunked:
        raise ValueError("chunked transfer encoding is not supported")
    if content_length is None:
        raise ValueError("Content-Length required")
    if not content_length.isascii() or not content_length.isdigit():
        raise ValueError("Content-Length must be an ASCII integer")
    if len(content_length) > MAX_CONTENT_LENGTH_DIGITS:
        raise ValueError(f"Content-Length has too many digits (> {MAX_CONTENT_LENGTH_DIGITS})")
    length = int(content_length)
    if not (1 <= length <= MAX_REQUEST_BYTES):
        raise ValueError(f"Content-Length must be 1..{MAX_REQUEST_BYTES}")


def validate_base_url(url: str) -> None:
    """Validate that *url* is a loopback HTTP origin.

    Raises ``ValueError`` if it is not.
    """
    from urllib.parse import urlsplit
    parsed = urlsplit(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"scheme must be one of {ALLOWED_SCHEMES}")
    if parsed.hostname not in ALLOWED_HOSTNAMES:
        raise ValueError(f"hostname must be one of {ALLOWED_HOSTNAMES}")
    if parsed.path not in ("", "/"):
        raise ValueError("path must be empty or /")
    if parsed.query or parsed.fragment:
        raise ValueError("query and fragment are not allowed")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("userinfo is not allowed")


def validate_state_fields(state: Any) -> None:
    """Validate workflow state field categories.

    Raises ``ValueError`` if any field is in an unexpected category.
    """
    if not isinstance(state, dict):
        raise ValueError("state must be a dict for field validation")
    known = STATE_BOOLS | STATE_COUNTS | STATE_TEXTS | STATE_VALUES
    extra = set(state) - known
    if extra:
        raise ValueError(f"unknown state field(s): {extra}")
    for key, value in state.items():
        if key in STATE_BOOLS:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
        elif key in STATE_COUNTS:
            if not (isinstance(value, int) and 0 <= value <= 1_000_000):
                raise ValueError(f"invalid counter: {key}")
        elif key in STATE_TEXTS:
            if not (isinstance(value, str) and len(value) <= 512):
                raise ValueError(f"invalid text: {key}")


def all_question_names() -> List[str]:
    """Return the full list of question names (12 capability + 2 legacy)."""
    return list(ALL_QUESTION_NAMES)
