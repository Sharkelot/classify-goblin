# Offline multimodal benchmark (JEV-MM-15)

A small, auditable, fully deterministic fixture set for multimodal evidence
decisions and profile recommendations.  It is a **fixture set only**: it
exercises the schema, routing, abstention, and fail-closed behavior of the
deterministic policy.  It is **not** a model-quality benchmark and it
produces **no quality claim** — a capability claim requires held-out
support that passes the acceptance gates.

## What it covers

* **Modalities:** `code`, `pdf`, `image`, `video`, `audio`
* **Domains:** `code_review`, `document_qa`, `screen_understanding`,
  `profile_recommendation`
* **Scenario classes:** `clean`, `ambiguous`, `adversarial`, `unavailable`
* **Questions:** `evidence_next_hand`, `outcome`, `needs_review`,
  `profile_fit`, `tool_screening`

The example count is
`len(modalities) * len(domains) * len(scenario_classes) * variants_per_group`.

## Splits and leakage

Examples are split into `train`, `validation`, `test`, and `calibration`
(600 examples at the default 5 variants).  `check_multimodal_leakage`
verifies no example id appears in more than one split, and
`calibration_gate` verifies the calibration split is non-empty and disjoint
from the test split.

## Regenerate the benchmark + manifest

```bash
python -m jev_laya_free.trainer multimodal-benchmark --seed 20260921
```

Writes the four split JSONL files and a provenance manifest under
`data/multimodal-benchmark/`.  The manifest records the seed, example
count, per-split support, the source/exclusion/capability states, and the
file SHA-256 hashes.  The output is a pure function of the seed: a second
run with the same seed produces byte-identical content.

## Offline smoke benchmark (no model, service, or network)

```bash
python -m jev_laya_free.trainer multimodal-smoke --output-dir reports/multimodal-smoke
```

Runs the full offline pipeline — build, split, leakage check, coverage,
raw + calibrated metrics, selective-risk curve, and the fail-closed
acceptance gate report — and writes a single verifiable report JSON plus
the four split files.  No model, service, or network is required; the
report is a pure function of the seed and carries the file SHA-256 hashes
for verification.

## Acceptance gates

The acceptance report is fail-closed.  A gate passes only when:

1. the modality is **verified by the local probe** (endpoint presence alone
   never counts — `source` must be `local_probe` and `state` must be
   `verified`);
2. the held-out split has **sufficient support**; and
3. the deterministic policy's internal-consistency metric (macro F1) clears
   the gate threshold.

Gates for `code`, `pdf`, and `image` are always present; gates for
`video` and `audio` are present only when the local probe verified them.
The `unavailable_backend_fail_closed` gate fails when any record has an
unavailable backend (the deterministic guard fails closed with no fallback
decision).  When evidence or support is missing, the required gates fail —
there is no silent pass on an empty or under-supported split.

## Hard-negative confusions

The fixture set carries hard-negative pairs so the confusions are
detectable:

* **extract-vs-render** (PDF): `extract_pdf_text` vs `render_pdf_page`
* **safe-vs-suspicious** (tool screening): `safe` vs `suspicious`
* **profile-fit vs invalid profile**: a predicted profile where the gold is
  `abstain`

Flipping the gold label on these records drops the macro F1 below the gate
threshold, so the corresponding gate fails — the confusion is caught by the
deterministic policy's internal-consistency metric.

## Raw vs calibrated metrics

The smoke report carries two distinct metric blocks:

* **raw** — the un-rescaled probability targets;
* **calibrated** — the targets after post-hoc temperature rescaling
  (`rescale_by_temperature`, temperature 2.0).

The two are never conflated: the report keeps them as separate blocks, and
the calibrated block is computed from a rescaled copy (the raw records are
not mutated).

## Invariant

No quality claim is produced from this fixture set.  A capability claim
requires held-out support that passes the acceptance gates, and a gate only
passes when the modality is verified by the local probe (never by endpoint
presence alone), the split has sufficient support, and the deterministic
policy's metric clears the threshold.
