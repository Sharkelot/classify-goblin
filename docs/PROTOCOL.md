# classify-goblin Typed-Decision Protocol

A minimal, dependency-free wire contract for a loopback typed-decision
service. It is intentionally small so that a third party can implement an
independent client against this document alone, and verify that their
implementation is compatible with the reference service.

**Scope.** The protocol defines the HTTP request/response shape, the
question and answer schemas, the error envelope, and the transport
constraints. It does **not** define inference quality, model identity, or
domain semantics — a backend may answer with any deterministic or learned
model, as long as the answers conform to the schema.

**Single source of truth.** The machine-readable constants and validators
live in [`src/classify_goblin/protocol.py`](../src/classify_goblin/protocol.py).
If this document and that module disagree, the module wins. The module is
exercised by `tests/test_protocol_spec.py`.

## Transport

| Property | Value |
|---|---|
| Scheme | `http` only (no TLS, no `https`) |
| Host | `127.0.0.1` or `localhost` (loopback only) |
| Path | `/v1/systemone` (the endpoint) |
| Method | `POST` |
| Content-Type (request) | `application/json` (exact) |
| Content-Type (response) | `application/json` (exact) |
| Request body size | 1 .. 65 536 bytes |
| Response body size | 1 .. 1 048 576 bytes |
| Auth | Optional `Authorization: Bearer <token>` |
| Redirects | Not followed (a redirect is an error) |
| Retries | Client-side, 0..5, on `429, 502, 503, 504, 529` |
| Timeout | 0 < timeout ≤ 60 seconds |

The client **must** reject any base URL that is not a loopback HTTP
origin. The server **must** bind to a loopback address.

## Request

```json
{
  "model": "local-rules-v1",
  "state": { "modality": "code" },
  "questions": {
    "model_routing": {
      "type": "choice",
      "instructions": "Which execution tier fits this turn?",
      "options": ["deterministic-code", "cheap-LLM", "frontier-LLM", "human"]
    }
  }
}
```

| Field | Type | Required | Constraints |
|---|---|---|---|
| `model` | string | yes | 1..128 chars, non-empty |
| `state` | string / object / array | yes | any JSON value; ≤ 32 768 bytes serialized |
| `questions` | object | yes | 1..32 entries |

Unknown top-level fields are rejected (`422`). The `model` value must be
`local-default` or the server's declared backend model name, otherwise
`422 unknown local model`.

### Questions

Each question is an object with a `type` of `choice`, `score`, or `noul`.

| Field | Applies to | Constraints |
|---|---|---|
| `type` | all | one of `choice`, `score`, `noul` |
| `instructions` | all | required; string/object/array ≤ 4 096 bytes |
| `options` | choice | list of strings, or dict of name→description; 1..255 entries |
| `criteria` | score | list of 2..10 level descriptions |
| `criteria` | noul | optional dict with keys ⊆ `{true, false}` |
| `criteria` | choice | (alias for `options`; exactly one of `options`/`criteria`) |

**Choice** requires exactly one of `options` (list or dict) or
`criteria` (dict). A list of options is normalized to a dict whose keys
are the option names.

**Score** does not accept `options`. `criteria` is a list of 2..10
descriptions, one per score level (0-indexed).

**Noul** does not accept `options`. `criteria` is an optional dict
mapping `true`/`false` to descriptions.

### Normalization

The server normalizes questions before inference:

- `choice`: `options` (list or dict) → `criteria` (dict)
- `score`: unchanged
- `noul`: unchanged

The client must apply the same normalization before validating the
response, so that answer-key lookups use the canonical `criteria` dict.

## CG-MM-14 — per-capability typed evidence questions

The workflow now sends **per-capability** typed questions instead of a
single undifferentiated global `next_hand`. `taxonomy.evidence_questions(modality)`
returns, for each implemented capability, an `evidence_next_hand` choice
whose `criteria` are that capability's **ordered evidence-step labels**
plus a `needs_review` noul. The labels are stable and ordered; the
broken global 8-option `next_hand` is never reused. Unknown modalities
fail closed.

`taxonomy.profile_fit_question(ordered_assignees)` returns a choice whose
`criteria` are the roster's ordered valid assignees followed by
`abstain`. It is **advisory only**: it suggests a profile but never
authorizes an assignment — the deterministic assessor and revalidation
own that. Empty or duplicate rosters fail closed.

The workflow's modality validation is derived from a single source of
truth, `taxonomy.EVIDENCE_HANDS` (the empty/text path plus every
capability with an implemented evidence path); an unknown modality fails
closed. The Qwen system prompt carries explicit `Capability under
evaluation:` and (when present) `Task identity:` lines.

The text-only `/v1/systemone` wire contract is unchanged: the new typed
questions round-trip through the reference service with label order
preserved, and a non-conforming model answer still fails closed
(502/503, no fallback). Model recommendations may suggest an evidence
hand or a profile but can never authorize a tool, write, retry, stop, or
assignment.

## Response

```json
{
  "model": "local-rules-v1",
  "request_id": "14d89923-…",
  "answers": {
    "model_routing": {
      "type": "choice",
      "choice": "deterministic-code",
      "confidence": 0.75,
      "probabilities": {
        "deterministic-code": 0.75,
        "cheap-LLM": 0.10,
        "frontier-LLM": 0.10,
        "human": 0.05
      }
    }
  },
  "usage": { "input_tokens": 0, "output_tokens": 0 }
}
```

| Field | Type | Required | Constraints |
|---|---|---|---|
| `model` | string | yes | non-empty |
| `request_id` | string | no | non-empty if present |
| `answers` | object | yes | keys must exactly match the request question names |
| `usage` | object | yes | `input_tokens` and `output_tokens` are non-negative ints |

### Answers

The `answers` object must have **exactly** the same keys as the request
`questions` object. Each answer's `type` must match the corresponding
question's `type`.

| Answer type | Required fields | Constraints |
|---|---|---|
| `noul` | `noul` | 0 ≤ noul ≤ 1 |
| `choice` | `choice`, `confidence`, `probabilities` | `choice` must be a key in `probabilities`; `probabilities[choice]` must be the max; `confidence` 0..1; `probabilities` sum to 1 ± 0.002 |
| `score` | `score`, `confidence`, `probabilities`, `legend` | `score` 0..(levels−1); `score` ≈ weighted mean ± 0.02; `legend` maps index→level description; `probabilities` sum to 1 ± 0.002 |

All probabilities are in `[0, 1]` and must sum to 1 (tolerance 0.002).
Confidence is in `[0, 1]`.

## Error envelope

Non-200 responses use a uniform JSON envelope:

```json
{
  "error": { "code": 422, "message": "unknown question field(s): {'foo'}" },
  "request_id": "14d89923-…"
}
```

| Status | Meaning |
|---|---|
| 400 | Malformed HTTP (bad Content-Length, chunked body, etc.) |
| 401 | Missing or wrong bearer token |
| 404 | Unknown path |
| 408 | Request read timed out |
| 413 | Body outside 1..65 536 bytes |
| 415 | Content-Type is not `application/json` |
| 422 | Schema validation failure or unknown model |
| 502 | Backend rejected the request or returned invalid output |
| 503 | Backend unavailable (generic) |
| 504 | Backend timed out |
| 529 | Backend busy (inference lock held) |

Statuses `429, 502, 503, 504, 529` are **retryable** by the client.

## Reference service

The reference implementation lives in
[`src/classify_goblin/reference_service.py`](../src/classify_goblin/reference_service.py).

```bash
# Start (loopback only, port 8093, no auth)
PYTHONPATH=src python -m classify_goblin.reference_service --port 8093

# With bearer-token auth
CLASSIFY_GOBLIN_LOCAL_API_KEY=secret PYTHONPATH=src python -m classify_goblin.reference_service --port 8093

# Health check
curl http://127.0.0.1:8093/health
```

The reference service uses a deterministic lexical-overlap backend (the
same algorithm as the existing `RulesBackend`). It is not a learned
model; it exists to prove the wire contract is implementable.

## Compatibility checklist

A third-party client is **compatible** with this protocol if and only if:

1. It sends `POST /v1/systemone` with `Content-Type: application/json` to a
   loopback HTTP origin.
2. It sends a JSON body with `model`, `state`, and `questions` fields that
   pass `protocol.validate_request`.
3. It reads the response body, parses JSON, and validates it with
   `protocol.validate_response` (using normalized questions).
4. It raises an error (not a fallback answer) on any transport or schema
   failure.
5. It retries only on `429, 502, 503, 504, 529`, at most 5 times.
6. It rejects non-loopback base URLs at construction time.

The reference client in
[`src/classify_goblin/protocol_client.py`](../src/classify_goblin/protocol_client.py)
satisfies all six. `tests/test_reference_service.py` exercises the full
round-trip against the reference service.
