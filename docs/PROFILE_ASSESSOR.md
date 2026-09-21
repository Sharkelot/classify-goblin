# Hermes Profile Assessor (JEV-MM-11)

Advisory-only profile fit assessment for Kanban task routing. The assessor
reads a sanitized profile roster snapshot, applies deterministic
eligibility filters, scores fit/confidence with a transparent heuristic,
and returns a verdict. **It never creates, assigns, or runs a task.**
The dispatcher revalidates the selected profile and owns every side effect.

## What it does

1. **Sanitized roster snapshot** — `RosterSnapshot.from_dict` or
   `parse_profile_dir` / `parse_profiles_root`. The parser opens ONLY
   `profile.yaml`, `config.yaml`, and the `skills/` directory listing.
   It never touches `.env`, `auth.json`, `memories/`, `sessions/`, or
   any other file. No bytes, raw reasoning, or secret values enter an
   entry.

2. **Deterministic filters F1-F7** (fail-closed, fixed order):

   | # | Filter | Reason code |
   |---|--------|-------------|
   | F1 | profile on disk | `not_on_disk` |
   | F2 | valid Kanban assignee | `not_valid_assignee` |
   | F3 | context >= min_context (unknown = fail) | `context_below_min` / `context_unknown` |
   | F4 | all required modalities declared | `modality_not_declared` |
   | F5 | all required capabilities declared | `capability_not_declared` |
   | F6 | workspace matches (unknown = pass + penalty) | `workspace_mismatch` |
   | F7 | status not blocked/unavailable | `status_blocked` / `status_unavailable` |

   The prefix property holds: a candidate excluded by the first k filters
   is also excluded by the full chain.

3. **Advisory fit scoring** (transparent heuristic, NOT a learned model):

   - Fit = 0.40·capability + 0.20·modality + 0.15·context + 0.10·description + 0.15·skill, clamped to [0, 1].
   - Confidence = 1.0 − penalties, clamped to [0, 1]:
     - description_auto: −0.25
     - status unknown/blocked: −0.15
     - workspace unknown: −0.15
     - context unknown: −0.10
     - snapshot age > 600s: −0.10

4. **Verdict** (deterministic, in contract order):
   - `abstain` / `no_eligible_candidate` — no candidate survived F1-F7.
   - `review` / `high_risk_task` — request risk is `high`.
   - `review` / `ambiguous_candidates` — top two within ε=0.06.
   - `review` / `low_confidence` — top confidence < 0.40.
   - `review` / `stale_snapshot` — snapshot age > 300s.
   - `recommend` — otherwise.

5. **Dispatcher revalidation** — `revalidate(name, fresh_snapshot,
   expected_digest)` returns `ok` / `stale` / `missing` / `invalid`.
   `first_valid_candidate` walks the ranked list and picks the first
   profile that revalidates to `ok`.

## Usage

### Module API

```python
from jev_laya_free.profile_assessor import (
    RosterSnapshot, AssessmentRequest, assess, revalidate)

snapshot = RosterSnapshot.from_dict(json.load(open('roster.json')))
request = AssessmentRequest.from_dict(json.load(open('request.json')))
result = assess(request, snapshot)
# result['verdict'] == 'recommend' | 'review' | 'abstain'
# result['candidates'] is ranked, deterministic, JSON-serializable.
```

### CLI

```bash
# Roster + request -> assessment
python -m jev_laya_free.profile_assessor.cli \
    --roster roster.json --request request.json

# Parse profile dirs directly (no roster file)
python -m jev_laya_free.profile_assessor.cli --profiles-root DIR
```

Exit codes: 0 = success, 2 = missing/invalid input.

## Request schema

```json
{
  "task_kind": "code",
  "required_modalities": ["text", "vision"],
  "min_context": 0,
  "required_capabilities": ["code-review"],
  "workspace_repo": null,
  "risk": "low",
  "urgency": "normal",
  "cost_hint": "low"
}
```

All fields except `task_kind` are optional with safe defaults.
`risk` ∈ {low, medium, high}; `urgency` ∈ {normal, high};
`cost_hint` ∈ {low, medium, high}.

## Roster schema

```json
{
  "snapshot_time": "2026-09-21T00:00:00Z",
  "valid_assignees": ["developer-goblin", ...],
  "entries": [
    {
      "name": "developer-goblin",
      "on_disk": true,
      "valid_assignee": true,
      "description": "coding agent",
      "description_auto": false,
      "model_label": "Qwen3.8-MXFP4",
      "provider_label": "qwen-27b",
      "context_length": 200000,
      "modalities": ["text", "vision"],
      "skill_names": ["autonomous-ai-agents"],
      "capabilities": ["code", "code-review"],
      "workspace": null,
      "status": "ok",
      "snapshot_time": "2026-09-21T00:00:00Z"
    }
  ]
}
```

`digest` is optional; if absent it is computed from the payload.

## Output schema

```json
{
  "verdict": "recommend",
  "reason": null,
  "candidates": [
    {
      "profile": "developer-goblin",
      "eligible": true,
      "fit": 0.90,
      "confidence": 0.85,
      "evidence": [],
      "entry_digest": "abc123...",
      "rank": 1
    }
  ],
  "excluded": [
    {"profile": "luna-goblin", "filter_reason": "modality_not_declared"}
  ],
  "advisory_only": true,
  "roster_digest": "def456...",
  "request_digest": "789abc..."
}
```

## Safety invariants

- **Advisory only**: `advisory_only` is always `true`. No model output
  authorizes execution or delegation.
- **No secrets**: the parser reads only allowlisted files. No `.env`,
  `auth.json`, `memories/`, or `sessions/` content enters the output.
- **No bytes**: all output is JSON-serializable strings, ints, bools,
  lists, and dicts. No raw bytes.
- **Deterministic**: the same roster + request always produces the same
  output. Ties break by name ascending.
- **Fail-closed**: unknown values exclude the candidate with a fixed
  reason code, never a silent pass.
- **Revalidation gate (JEV-MM-14)**: `workflow.validate_profile_selection(
  assessment, fresh_snapshot)` is the deterministic gate a caller must
  pass before invoking Hermes Kanban. Only a `recommend` verdict is
  eligible; the selected profile is the first ranked candidate that
  revalidates to `ok` against a fresh roster snapshot. A stale, missing,
  or invalid candidate — or any non-recommend verdict — fails closed
  with `None` (no fallback). The assessor's recommendation is advisory;
  this revalidation, not the model, owns the assignment.

## Tests

- `tests/test_profile_assessor_unit.py` — 33 unit tests covering
  filters F1-F7, verdict rules, tie-breaking, revalidation, the pure
  parser, and output hygiene.
- `tests/test_profile_assessor_integration.py` — 11 integration tests
  covering the CLI end-to-end, error paths, and digest consistency.

Run: `PYTHONPATH=src python3 -m pytest tests/test_profile_assessor_unit.py tests/test_profile_assessor_integration.py -q`
