# jev-laya-free

The advisory catalog now covers the 12 awesome-jev capability families (routing, action
confidence, screening, progress, completion, skills, compaction, citations, RAG, semantic
find, composite scoring, and intent routing). See `docs/jev-hermes-integration.md` and
`reports/final-capabilities.json`; deterministic workflow guards remain authoritative.

An independent, MIT-licensed local implementation of the documented Jev typed-decision
HTTP shape, with a Python client and optional Laya inference. This is **not TypeSafe's
hosted model**, an official TypeSafe SDK, a reproduction of Jev's weights, or a promise
of equivalent predictions. Local confidence is **not calibrated Jev confidence**.

No hosted API, model weights, private traces, or secrets are included. The default backend
is a deterministic lexical demo for testing. It does not understand arbitrary instructions
and must not be used as an intelligent safety classifier. The optional Laya backend is
advisory; its reliability must be evaluated on your own data. “Free” refers to this source
and local operation without a hosted API fee; model licensing and hardware costs are separate.

## Repository status (2026-09-21)

The local artifact broker, typed protocol, deterministic workflow guard, profile assessor,
and offline benchmark all existed before the Qwen backend was added. What was **unavailable
until the backend path was verified** is model serving for the multimodal modalities: the
Qwen artifact backend (`--backend qwen`) was not live-verified until JEV-MM-17 (2026-09-21),
which started an isolated shadow process on loopback `127.0.0.1:8094` against the local
Qwen endpoint and confirmed the typed contract end-to-end (18/18 smoke cases). The live
`127.0.0.1:8093` DistilBERT service and the `127.0.0.1:8091` Laya sidecar were left
untouched throughout. See `reports/jev-mm17-shadow-verification.md` for the full evidence.

The modality matrix below is from the actual runtime probe (JEV-MM-01 historical +
JEV-MM-13 + JEV-MM-17 fresh, all against the live loopback Qwen endpoint). It is
**probe evidence, not an upstream claim**: a modality is "verified" only when the local
probe exercised it and observed a successful decode.

| Modality | Probe state | Backend support | Evidence |
|----------|-------------|----------------|----------|
| text     | verified    | supported      | JEV-MM-17 typed 200 |
| image    | verified    | supported      | JEV-MM-01 + JEV-MM-17 (image tokens accounted) |
| video    | verified    | supported      | JEV-MM-13 (`video_url` data URI, 200, `multimodal_tokens.video`) |
| code     | (new)       | supported      | JEV-MM-17 typed 200 (choice-only through full server) |
| pdf      | unavailable | supported (declared; probe did not verify) | JEV-MM-12 declared; live verification is a future probe run |
| audio    | unavailable | fail-closed adapter (`audio_not_verified`) | JEV-MM-13 (HTTP 400 "At most 0 audio(s)") |

`pdf` is declared supported by the backend but its live probe verification is the
responsibility of a future probe run. `audio` is not supported by the endpoint (no
transcription route), so the audio adapter is fail-closed: it returns
`audio_not_verified` rather than a fake success. Unsupported (unknown) modalities fail
closed with a stable 503 — never a fallback to text.

## Start locally

Python 3.10+; the base package uses only the standard library. From this directory:

```bash
PYTHONPATH=src python3 -m jev_laya_free.server --backend rules --port 8093
```

In another terminal:

```bash
PYTHONPATH=src python3 examples/client.py
curl --fail-with-body http://127.0.0.1:8093/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @examples/request.json
```

Optional installation: `python3 -m pip install .`; then run `jev-laya-free`. Building requires
setuptools. For a prepared offline build environment, use `--no-build-isolation --no-deps`.
The server is loopback-only, defaults to port 8093, and never starts or restarts other services.
Set `JEV_LOCAL_API_KEY` in the server and client environment to enable optional bearer auth.
The client also accepts `api_key=...`. Never commit a real key. With no key configured,
local callers are unauthenticated. This standard-library server is for local development,
not an Internet-facing production service.

## Wire contract and client

`POST /v1/systemone` accepts `state`, `questions`, and required `model`. State may be a string,
object, or list containing JSON values. Each question keeps its caller-supplied name.

| Type | Input | Answer |
| --- | --- | --- |
| `choice` | `options` list/map, or `criteria` list/map | `type`, `choice`, full `probabilities`, `confidence` |
| `score` | ordered `criteria` list | `type`, numeric `score`, indexed `probabilities`, `legend`, `confidence` |
| `noul` | optional `criteria` with `true`/`false` descriptions | `type`, `noul` probability only |

Every question requires `instructions`, which may be text, an object, or a list. Criterion
values may also be structured; Choice additionally accepts null descriptions. Choice list
entries must be unique strings. Supplying both `options` and `criteria` is rejected.
Score is the probability-weighted ordinal index, not a normalized 0–1 score.
Structured Score descriptions are serialized to compact JSON strings in the legend.

Responses include `model`, `answers`, `usage`, and a locally generated UUID `request_id`.
Models are honestly named `local-rules-v1` or `local-laya`; `local-default` selects the
configured backend. Hosted Jev model IDs are rejected, never silently mapped to Laya.
Rules usage is zero because no tokenizer/model runs. Laya usage is reported by its backend;
those counters are not directly comparable to Jev billing tokens. No cost/provider claims
are emitted.

```python
from jev_laya_free import TypeSafeClient, Choice, Score, Noul

with TypeSafeClient(timeout=5, retries=2) as client:
    result = client.system_one(
        state={"summary": "code test incomplete"},
        questions={
            "route": Choice(instructions="Evidence type?", options=["code", "pdf"]),
            "quality": Score(instructions="Quality?", criteria=["incomplete", "complete"]),
            "ready": Noul(instructions="Ready?"),
        },
    )
    print(result.answers["route"].choice)
```

`systemOne` is an alias of `system_one` with the same keyword arguments. Builders return
plain dictionaries; results are dictionaries with convenience attribute access. Use bracket
access for keys colliding with dictionary methods. This is a small independent client under
`jev_laya_free`. A bundled `typesafe_sdk` shim also re-exports `Choice`, `Noul`, `Score`,
`TypeSafeClient`, `AsyncTypeSafeClient`, `ClientError`, and `ValidationError`. It supports the
official quickstart import/call shape but is independent code, not the official distribution.
Use a separate environment from the official `typesafe-sdk`: both provide the same import
package and must not be co-installed.

Responses expose `answers` and grouped `choices`, `scores`, and `nouls` mappings of question
names to answer objects. For example, `response.choices["route"].choice`. Grouped views are
computed properties and do not add fields to the underlying response dictionary or wire JSON.

```python
from typesafe_sdk import AsyncTypeSafeClient, Noul

async def inspect():
    async with AsyncTypeSafeClient() as client:
        response = await client.system_one(
            state={"summary": "synthetic evidence"},
            questions={"ready": Noul(instructions="Is the evidence complete?")},
        )
    return response.nouls["ready"].noul
```

The async facade runs the same local transport in `asyncio.to_thread`, preserving validation,
retries and timeouts without blocking the event loop. Cancelling the coroutine or leaving the
context does not cancel an in-flight worker/socket operation. No persistent HTTP session is held.

Client configuration precedence: explicit `base_url`, then `TYPESAFE_BASE_URL`, then
`http://127.0.0.1:8093`; explicit `api_key`, then `JEV_LOCAL_API_KEY`, then `TYPESAFE_API_KEY`.
Environment URLs still must be loopback HTTP. The server itself continues to use only
`JEV_LOCAL_API_KEY`; compatibility fallbacks are client-side. The client always supplies its
explicit `local-default` model default; `TYPESAFE_DEFAULT_MODEL` is intentionally ignored.
An explicit hosted model such as `jev-latest` receives an error, never a local substitution.

The client accepts only loopback HTTP origins, ignores proxy environment settings, refuses
redirects, and validates response types, names, distributions, scores and usage. It raises
`ClientError` on failed calls; it never fabricates a successful fallback decision. Invalid
local requests raise `ValidationError`. Timeouts default to 5 seconds per socket operation;
retries default to 2, with bounded exponential backoff for transport errors and HTTP
429/502/503/504/529. Validation and auth failures are not retried. The maximum is 5 retries.
A timeout does not cancel backend inference and is not a whole-call wall-clock deadline.

## Protocol

A minimal, dependency-free wire contract for a loopback typed-decision service, designed
so a third party can implement an independent client from the spec alone and verify
compatibility against the reference service.

- **Spec (machine-readable):** [`src/jev_laya_free/protocol.py`](src/jev_laya_free/protocol.py)
  — constants and validators; the single source of truth. Exercised by
  [`tests/test_protocol_spec.py`](tests/test_protocol_spec.py).
- **Human-readable spec:** [`docs/PROTOCOL.md`](docs/PROTOCOL.md) — transport, request,
  response, error envelope, and the compatibility checklist.
- **Reference service:** [`src/jev_laya_free/reference_service.py`](src/jev_laya_free/reference_service.py)
  — a stdlib-only HTTP server implementing the contract with a deterministic lexical
  backend (not a learned model). Start it with
  `PYTHONPATH=src python -m jev_laya_free.reference_service --port 8093`.
- **Independent client:** [`src/jev_laya_free/protocol_client.py`](src/jev_laya_free/protocol_client.py)
  — written against `protocol.py` only (no import from the reference service), stdlib-only.
  See [`docs/INDEPENDENT_CLIENT.md`](docs/INDEPENDENT_CLIENT.md).
- **Compatibility tests:** [`tests/test_reference_service.py`](tests/test_reference_service.py)
  — full round-trip, error paths, retry behavior, and transport constraints, run against
  the reference service on a random loopback port.

The protocol reuses the same question/answer schemas, bounds, and error semantics as the
existing wire contract above, but is defined in a single self-contained module so it can be
implemented and verified independently. The reference service's deterministic backend is a
contract proof, not a quality claim.

## Optional local Laya backend

Install/provide Laya and its compatible ML dependencies separately; this repository neither
vendors them nor downloads weights. Set `JEV_LAYA_MODEL_PATH` to an existing local checkpoint
directory. For example, if an external Laya checkout is already on disk:

```bash
export JEV_LAYA_MODEL_PATH=/path/to/existing/laya-checkpoint
export JEV_LAYA_DEVICE=cpu
PYTHONPATH=src:/path/to/existing/laya-source python3 -m jev_laya_free.server --backend laya
```

Use the Python environment containing the required PyTorch/Transformers/Laya dependencies.
The loader sets Hub and Transformers offline mode before importing Laya. Missing files fail
startup; there is no fallback download or substitution of the rules backend. The real model
was not loaded in this repository's offline tests; a fake Laya module tests translation and
normalization. External model/code licenses apply separately from this repository's MIT license.

The adapter translates options to Laya criteria, preserves optional Noul criteria, serializes
structured instructions, and strips Laya action metadata and Noul confidence. It caps Laya
calls at 8 questions and 8 Choice options. Laya's own tokenizer/context/head limits remain;
long input can be truncated upstream and head overflow can fail. HTTP byte limits are not a
promise that all evidence reaches the model. Keep summaries short and verify model behavior
before relying on any probability.

## Deterministic Hermes/Qwen evidence adapter

`jev_laya_free.workflow.decide(state, client=None)` is separate from the generic HTTP contract.
It is stateless and has no dependency on a private Hermes installation. Run the synthetic
example with `PYTHONPATH=src python3 examples/workflow.py`.

It hashes full JSON values before any model call. Terminal state wins over all other rules.
Without progress, three repeated actions, two context compactions, or three repeated tool
failures stop work; two identical actions request review. Artifact/board digest or result
changes count as progress. Caller-provided progress flags are also accepted. Callers must
supply trustworthy counters and digests; this adapter cannot verify those assertions.

A blocked gate suppresses tool routing and skips inference. Allowed code work selects source
inspection or a focused test, PDF work selects text extraction or selected-page rendering,
and image work selects Qwen vision inspection. Synthesis requires sufficient/grounded flags,
a source digest, and a location. **Qwen remains the analyzer and tool user.** All route names
are hints for the caller, not executable tool requests. Review/stop/terminal handling and
exactly-once terminal writes belong to the caller.

Optional model answers are returned separately as `advisory`, always with `authoritative=false`.
They cannot modify either the gate or route. Only a small metadata summary is sent in this
workflow mode, never action/result text or source bytes. The generic HTTP endpoint, by contrast,
passes supplied state to the local backend; it is not a credential scrubber. Invalid workflow
state raises instead of allowing work. A client outage preserves the deterministic decision.
This adapter does not configure Hermes, Qwen, DFlash, gateways, or any existing sidecar.

## Local Qwen image/PDF/code typed backend

`jev_laya_free.multimodal.qwen_service.QwenArtifactBackend` is a local-only,
fail-closed artifact backend: verified in-memory `ResolvedArtifact` bytes
(image/PDF/code) are converted into a typed JSON decision via the loopback
OpenAI-compatible Qwen endpoint, validated through `schema.answers`. It is wired
through `predict_artifacts`, so text-only requests keep their existing behavior.
Start it with `PYTHONPATH=src python -m jev_laya_free.server --backend qwen`
(endpoint from `JEV_QWEN_BASE_URL`, default `http://127.0.0.1:8080`; credential
from `JEV_QWEN_API_KEY`; timeout from `JEV_QWEN_TIMEOUT`). The backend declares
`SUPPORTED = {image, pdf, code, video}`; `audio` is a fail-closed adapter
(`audio_not_verified`) and unknown modalities fail closed with a stable 503 —
never a fallback to text. Usage is sanitized to artifact id/pages/truncation plus
token counts; no raw bytes, paths, credentials, or provider error bodies cross the
wire. See [`docs/QWEN_BACKEND.md`](docs/QWEN_BACKEND.md) for the verified vs.
unavailable modality table and the live probe result.

## Hermes profile assessor (advisory only)

`jev_laya_free.profile_assessor` is an **advisory-only** profile-fit assessment
for Kanban task routing. It reads a sanitized profile roster snapshot, applies
deterministic eligibility filters (F1–F7, fail-closed, fixed order), scores
fit/confidence with a transparent heuristic (not a learned model), and returns a
verdict. **It never creates, assigns, or runs a task** — the dispatcher
revalidates the selected profile and owns every side effect.

- **Sanitized metadata:** the roster parser opens only `profile.yaml`,
  `config.yaml`, and the `skills/` listing. It never touches `.env`, `auth.json`,
  `memories/`, or `sessions/`. No bytes, raw reasoning, or secret values enter an
  entry; all output is JSON-serializable.
- **Deterministic filters F1–F7:** profile on disk, valid assignee, context ≥
  min (unknown = fail), required modalities declared, required capabilities
  declared, workspace match (unknown = pass + penalty), status not
  blocked/unavailable. The prefix property holds.
- **Abstention and review:** `abstain` when no candidate survives; `review` for
  high-risk tasks, ambiguous top-two (ε=0.06), low confidence (<0.40), or stale
  snapshots (>300 s).
- **Revalidation gate (JEV-MM-14):** `workflow.validate_profile_selection(
  assessment, fresh_snapshot)` is the deterministic gate a caller must pass before
  invoking Hermes Kanban. Only a `recommend` verdict is eligible; the selected
  profile is the first ranked candidate that revalidates to `ok` against a fresh
  roster snapshot. A stale, missing, or invalid candidate — or any non-recommend
  verdict — fails closed with `None` (no fallback). The assessor's recommendation
  is advisory; this revalidation, not the model, owns the assignment.
- **No-side-effect delegation boundary:** the assessor is a pure function (same
  roster + request always produces the same output; ties break by name
  ascending). No model output authorizes execution or delegation.

Full contract, schemas, and usage: [`docs/PROFILE_ASSESSOR.md`](docs/PROFILE_ASSESSOR.md).

## Optional local DistilBERT training

Training is **not part of this release**. This package ships the inference runtime, the
typed protocol, artifact security, and the benchmark conformance path. The reproducible
training path for a DistilBERT-base encoder (`distilbert/distilbert-base-uncased`,
Apache-2.0, about 67M parameters) with a compact shared typed-decision head lives in the
**trainer** module. The base install stays standard-library only; the trainer owns its own
PyTorch/Transformers pins, the clean-venv setup script, and the training/test data.

To train, install the trainer in a separate environment (never the shared ComfyUI
environment or a global site-packages) and follow its pinned requirements. After training,
publish a checksummed checkpoint manifest and install it with:

```bash
goblin-jev download-checkpoint --manifest <manifest.json> --cache-dir <dir>
```

The local DistilBERT backend loads that checkpoint for inference. The base package does not
download model weights or run training; it only loads a manifest-verified checkpoint.

Download public assets with the Hugging Face CLI (the commands write only to ignored paths):

```bash
export JEV_DATA_DIR="$PWD/datasets/local"
export JEV_MODEL_DIR="$PWD/models/local"
mkdir -p "$JEV_DATA_DIR" "$JEV_MODEL_DIR"
hf download LocalLLaMA/typed-decisions --type dataset --local-dir "$JEV_DATA_DIR/typed-decisions"
hf download distilbert/distilbert-base-uncased --type model --local-dir "$JEV_MODEL_DIR/distilbert-base-uncased"
```

Alternatively let `datasets` and Transformers populate their normal cache through the preparation
and training commands. A disconnected synthetic smoke path does not need either download:

```bash
python -m jev_laya_free.trainer prepare \
  --include-synthetic --synthetic-count 24 --output data/hybrid.jsonl \
  --validation-output data/validation.jsonl --no-local-traces
python -m jev_laya_free.trainer smoke --tiny-random --device auto
python -m jev_laya_free.trainer evaluate --output reports/synthetic.json
```

For public preparation, use the dataset loader after installing the extras:

```bash
python -m jev_laya_free.trainer prepare \
  --public-dataset "$JEV_DATA_DIR/typed-decisions" --public-config all --public-split train \
  --include-synthetic --output data/hybrid.jsonl --validation-output data/validation.jsonl

# Keep the public test cases out of training; the two outputs together contain the test split.
python -m jev_laya_free.trainer prepare \
  --public-dataset "$JEV_DATA_DIR/typed-decisions" --public-config all --public-split test \
  --no-local-traces --output data/public-test-a.jsonl \
  --validation-output data/public-test-b.jsonl
```

Preparation uses the default redacted Hermes path when it exists. Pass another local Hermes
JSONL with `--local-traces`, or add `--no-local-traces` for a fully synthetic/public run:

```bash
python -m jev_laya_free.trainer prepare \
  --public-dataset LocalLLaMA/typed-decisions --local-traces \
  /home/coreys/models/laya-sidecar/data/hermes-traces.jsonl \
  --output data/hybrid.jsonl --validation-output data/validation.jsonl
```

Local rows are redacted and bounded during normalization, but the resulting training files can
still contain private task context. Keep every data/output path private and ignored; no local
trace, report, dataset, checkpoint, `safetensors`, or binary weight is part of this repository.
The split is group-stable so rows sharing a task group do not cross the train/validation boundary.

Train and evaluate a checkpoint after the data and base model are available locally:

```bash
python -m jev_laya_free.trainer train \
  --data data/hybrid.jsonl --model "$JEV_MODEL_DIR/distilbert-base-uncased" \
  --validation-data data/validation.jsonl \
  --local-files-only --output-dir checkpoints/local-distilbert \
  --device cuda --precision bf16 --warmup-epochs 1 --epochs 3
python -m jev_laya_free.trainer evaluate \
  --predictions reports/predictions.jsonl --output reports/evaluation.json
```

The regular `smoke` command loads the real base checkpoint and performs one forward/backward
optimizer step. `--tiny-random` uses a one-layer random DistilBERT configuration and is the
offline CI path; it verifies tensor shapes, the soft-target/Brier/RPS loss, and optimizer wiring,
not model quality. The evaluator reports accuracy, NLL, Brier, ECE, RPS, temperature fitting,
repeat-without-progress recall, evidence-route accuracy, and deterministic-rule precedence.

After a checkpoint passes held-out acceptance checks, enable it explicitly:

```bash
export JEV_DISTILBERT_MODEL_PATH="$PWD/checkpoints/local-distilbert"
export JEV_DISTILBERT_DEVICE=cuda
PYTHONPATH=src python3 -m jev_laya_free.server --backend distilbert --port 8093
```

`distilbert` is opt-in and fail-closed: missing optional dependencies, missing local files, an
unsupported checkpoint, or an unknown question task abort backend startup. It never replaces the
rules backend automatically. The deterministic Hermes/Qwen workflow guard remains authoritative;
the DistilBERT probabilities cannot authorize retries, tool calls, gateway changes, or terminal
writes. Qwen remains the analyzer and tool user for code, PDF, and image evidence.

The checkpoint format is local (`jev_laya_config.json`, `encoder/`, `head.pt`, tokenizer files)
and intentionally separate from Laya weights. No public download or GPU run is performed during
package import or the base test suite.

### Current status (2026-09-20)

A checkpoint was trained on the RTX 4060 Ti (bf16, 1 warmup + 3 epochs) from the local
typed-decisions set (1035 train / 285 validation, local + public + synthetic) with the public
`test` split held out, and saved to `checkpoints/local-distilbert` (float16, 192-dim head).
Held-out evaluation (400 test examples, 2000 prediction records) reports validation accuracy 0.56,
ECE 0.0392, and deterministic-rule precedence 13/13. The advisory-only gates that have support
pass (calibration, deterministic guard precedence); the modality/risk gates report `support=0`
because the public test set carries no `loop_state`/`next_hand`/`high_risk_code`/`modality`
fields, which is a data-coverage gap, not a model-quality failure. No gate authorizes execution.
The full suite is green (84 passed, 2 skipped, 44 subtests) and the deterministic loop gate is
backend-independent.

The trained backend is deployed as the systemd user unit `jev-distilbert.service`
(loopback-only, `Restart=on-failure`) serving `127.0.0.1:8093` with
`JEV_DISTILBERT_MODEL_PATH=checkpoints/local-distilbert` and `JEV_DISTILBERT_DEVICE=cuda`.
The old Laya sidecar on `127.0.0.1:8091` (`laya-loop-gate.service`) is untouched.

Held-out temperature calibration (2026-09-20): the temperature was re-fitted on the
public test split (400 examples, 2000 records, never seen in training) via
`fit_temperature()` and the fit is stable (bootstrap mean 0.97, std 0.046, 100 resamples).
The fitted temperature is 1.0 (held-out NLL 1.0128 both before and after), so the model is
already well-calibrated and the advisory probabilities are unchanged by the calibration
pass; the deterministic guard precedence stays 13/13. The fitted temperature is stored in
`checkpoints/local-distilbert/jev_laya_config.json` (`calibration.temperature`) and the
serving backend divides logits by it before softmax (default 1.0 when absent, so older
checkpoints still load). The held-out before/after comparison, per-kind metrics, and the
bootstrap stability are recorded in `reports/calibration.json` (run via
`python -m jev_laya_free.trainer calibrate`).

### Capability benchmark data

The public typed-decisions dataset labels only the `progress` question, so the other 11
capability families (model routing, confidence action, tool screening, completion, skill
selection, compaction, citation, RAG filtering, semantic find, composite scoring, intent
routing) had no held-out labels. The trainer's capability-benchmark builder generates a
small, fully deterministic labeled fixture for every capability family, with a deterministic
three-way split (train / validation / held-out test) that has no state or question leakage
between splits.

Regeneration is a pure function of the seed: the same seed always produces byte-identical
JSONL, and no external service, network call, or secret is required.

```bash
# In the trainer environment (not this package):
# Smoke fixture (default): 12 capabilities x 7 scenarios x 5 variants = 420 examples.
python -m jev_laya_free.trainer capability-benchmark \
  --seed 20260920 --validation-fraction 0.2 --test-fraction 0.1 \
  --output-dir data/capability-benchmark

# Full fixture: 12 x 7 x 76 = 6384 examples total — 4476 train, 1272 validation, and
# a 636-example held-out test split (53 per capability). The 1908 figure is the
# combined validation+test support (159 per capability), not the test split size.
python -m jev_laya_free.trainer capability-benchmark \
  --seed 20260920 --variants 76 --output-dir data/capability-benchmark-full
```

Every gold label is **advisory**: the model output is a probability distribution over the
question's criteria, and deterministic code owns the thresholds and fail-open behavior (see
`taxonomy.CATALOG`). A gold label never turns a deterministic gate on or off by itself.
Labels are curated from the documented capability policy (the scenario matrix in
`capability_benchmark.GOLD`), not fabricated external Jev measurements; every example carries
`metadata.label_rationale` and `metadata.provenance` so a label can be audited without leaving
the repository.

The large split JSONL files are kept out of Git (regenerable from the seed); the checked-in
`capability-benchmark-manifest.json` and `capability-benchmark-sample.jsonl` for each fixture
record the coverage, file checksums, fixture class, and the regeneration command.

### Capability-benchmark checkpoint (2026-09-20)

A second checkpoint was trained on the RTX 4060 Ti (bf16, 1 warmup + 3 epochs) from the
**full** capability-benchmark train split (4476 train / 1272 validation) with the held-out
test split (636 examples, 53 per capability) never seen in training, and saved to
`checkpoints/capability-distilbert` (float16, 192-dim head). This checkpoint is the one
evaluated for the per-capability report, and it is a substantial improvement over the
previous hybrid-data checkpoint on the same held-out test split:

| Metric | previous (`local-distilbert`) | capability (`capability-distilbert`) | delta |
| --- | --- | --- | --- |
| accuracy | 0.1164 | 0.2830 | +0.1667 |
| NLL | 1.3419 | 0.9585 | -0.3834 |
| Brier | 0.8746 | 0.5901 | -0.2845 |
| ECE | 0.4533 | 0.2289 | -0.2244 |

Per-capability held-out accuracy (53 examples each, never seen in training):
citation 0.87, semantic_find 0.87, completion 1.00, model_routing 0.13, intent_routing 0.13,
rag_filter 0.13, composite 0.13, tool_screening 0.13, compaction 0.00, confidence_action 0.00,
skill_selection 0.00, progress 0.00. The single shared DistilBERT head learns binary
(noul) decisions well (citation, semantic_find, completion) but not multi-option routing
(choice/score types remain low) — a real model limitation, not a data gap. The only
regression versus the previous checkpoint is `progress` (0.87 → 0.00): the previous
checkpoint was tuned on the public typed-decisions set that labels only `progress`, so it
overfit that one family at the cost of the other eleven.

Held-out temperature calibration (2026-09-20): the temperature was re-fitted on the
capability-benchmark test split (636 examples, 53 per capability, never seen in training) via
`fit_temperature()` and the fit is stable (bootstrap 200 resamples, seed 20260920). Because
the temperature is fitted on the held-out test split itself, the checkpoint's held-out
metrics are post-hoc test calibration, not raw held-out quality — label them accordingly
when comparing checkpoints. The fitted temperature is
1.1875 (held-out NLL 0.9598 before → 0.9585 after). The fitted temperature is stored in
`checkpoints/capability-distilbert/jev_laya_config.json` (`calibration.temperature`) and the
serving backend divides logits by it before softmax (default 1.0 when absent, so older
checkpoints still load). The held-out before/after comparison, per-kind metrics, and the
bootstrap stability are recorded in `reports/calibration-capability.json` (run via
`python -m jev_laya_free.trainer calibrate` in the trainer environment).

`reports/final-capabilities.json` is **regenerated** (not hand-maintained) by the
trainer's `regenerate_final_capabilities` script, which loads both checkpoints, evaluates
each at
its own persisted serving temperature on the held-out test split, computes per-capability
accuracy/NLL/Brier/ECE/RPS, runs a 200-resample bootstrap for stability, and records the
deterministic guard precedence. The full regeneration command (train → calibrate → report)
is recorded in `reports/final-capabilities.json` under `regeneration_command`.

The `capability-distilbert` checkpoint weights (`head.pt`, tokenizer files, `evaluation.json`)
are kept out of Git (regenerable from the seed); only `jev_laya_config.json` is committed,
matching the `local-distilbert` convention. No Laya weights or secrets are committed.

## Bounds and failures

- Requests: 64 KiB, depth 16, 1–32 questions, question/option names up to 128 characters,
  4 KiB per instruction/description, 1–255 Choice options, 2–10 Score levels.
- Workflow state: 32 KiB, strict known fields, actual booleans and bounded nonnegative counters.
- Duplicate JSON keys, NaN/Infinity, unknown request/question fields, invalid distributions,
  mismatched answer names, and unsupported types are rejected. Rounding tolerance is 0.002
  for distribution sums and 0.02 for weighted Score consistency.
- One inference at a time; a busy backend returns 529. At most 16 request threads are active;
  excess connections close. Socket reads have a 5-second timeout. Inference itself has no hard
  deadline, so a hung backend requires operator intervention.
- HTTP 422 indicates invalid input/model; 401 auth failure; 413 size; 415 content type;
  400 framing; 408 read timeout; 502 invalid/rejected backend output; 503 backend exception.
  Error bodies contain no answers or internal exception text. Chunked requests are unsupported.
- The client bounds response reads to 1 MiB. No payload or credential access logging is enabled.

## Compatibility evidence and unresolved details

Checked against official documentation on 2026-09-20:
[API reference](https://docs.typesafe.ai/api),
[quickstart](https://docs.typesafe.ai/introduction/quickstart),
[Choice](https://docs.typesafe.ai/primitives/choice),
[Score](https://docs.typesafe.ai/primitives/score), and
[Noul](https://docs.typesafe.ai/primitives/noul).
No hosted requests or interoperability tests using the official SDK distribution were performed.
The local shim is tested against the published synchronous and asynchronous quickstart shapes.

The canonical reference specifies Choice `criteria`; `options` is an intentional local alias
for callers needing that shape. Raw HTTP requires `model`, as the official API does; local clients supply
`local-default` when no model argument is provided. `request_id` is a local extension, absent from the inspected
response schema. The reference accepts structured Score criteria but describes legend values
as strings; compact JSON serialization is our explicit resolution. The official minimum Choice
cardinality and maximum question count are not fully specified there; local bounds above are
implementation policy. Confidence calibration/formula, performance, context capacity, prediction
quality, error body details, model discovery, streaming, hosted authentication,
and official SDK features beyond the shim subset are outside the compatibility claim.

## Offline checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests examples
PYTHONPATH=src python3 examples/workflow.py
```

Tests create ephemeral loopback servers and use synthetic data, the rules backend, or mocks.
They exercise schemas, typed roundtrips, auth, failures, retries, Laya translation, and deterministic
safety precedence. No private traces, real model weights, hosted calls, or running gateways are needed.

## Benchmarks and reports

| Artifact | What it records |
| --- | --- |
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | How to run the offline smoke, rules-backend, and full held-out benchmarks, and how to interpret accuracy, ECE, NLL, Brier, support, latency, and guard metrics. |
| [docs/MULTIMODAL_BENCHMARK.md](docs/MULTIMODAL_BENCHMARK.md) | The offline multimodal fixture set (JEV-MM-15): modalities, splits, leakage checks, the fail-closed acceptance gates, hard-negative confusions, and raw vs. calibrated metrics. A fixture set, not a model-quality claim. |
| [docs/jev-hermes-integration.md](docs/jev-hermes-integration.md) | Architecture and integration: typed questions, DistilBERT advisory layer, deterministic guard, workflow adapter, loopback service. |
| [reports/final-capabilities.json](reports/final-capabilities.json) | Per-capability held-out accuracy/NLL/Brier/ECE/RPS for the full 12-capability catalog, previous-checkpoint comparison, bootstrap stability, and deterministic guard precedence (regenerated, not hand-maintained). |
| [reports/calibration-capability.json](reports/calibration-capability.json) | Held-out temperature fit (1.1875), before/after NLL, per-kind metrics, and bootstrap stability. |
| [reports/jev-comparison.md](reports/jev-comparison.md) / [reports/jev-comparison.json](reports/jev-comparison.json) | Direct, source-grounded Jev comparison with explicit unavailable-runtime boundary; no invented Jev quality numbers. |

The benchmark guide ([docs/BENCHMARK.md](docs/BENCHMARK.md)) documents the exact
`capability-benchmark` command for the full fixture (`--variants 76`; there is no `--full`
flag), the 636-example held-out test split (53 per capability; 1908 is the combined
validation+test support), and the post-hoc test-calibration caveat. Offline results report
schema coverage only; learned-model quality comes from the model evaluation scripts and must
label raw versus post-hoc calibration. Rules results report deterministic behavior and
transport/schema outcomes, not Jev-model accuracy.

### Compatibility matrix

| Surface | Local behavior | Boundary |
| --- | --- | --- |
| HTTP request | Required model/state/questions; all questions require instructions | Local model IDs only |
| Choice | Canonical criteria map plus options/list aliases | Aliases extend the documented reference |
| Score / Noul | Ordered levels / optional yes-no criteria | Local inference and calibration |
| HTTP response | model/answers/usage; UUID request_id | request_id is a local extension |
| Sync SDK | typesafe_sdk imports, context manager, system_one/systemOne | Builders are dictionaries, not official SDK model classes |
| Async SDK | AsyncTypeSafeClient, async context manager, await system_one/systemOne | Thread-backed transport; cancellation cannot stop inference |
| Response access | answers plus grouped choices/scores/nouls; mapping access | Views contain typed answer dictionaries with attribute access |
| Environment | TYPESAFE_BASE_URL and TYPESAFE_API_KEY fallbacks | Loopback enforced; local-default remains explicit |

The added SDK checks follow the [official Python quickstart shapes](https://docs.typesafe.ai/sdk/python)
using synthetic inputs against an ephemeral local rules server. They check both clients, grouped
views, environment precedence, required wire model, and hosted-model rejection.

### Local artifact reference contract

`system_one` / `systemOne` on both clients accept optional `artifacts`. Omitting
it preserves the existing wire request. Each reference contains `id` (opaque
ASCII letters, digits, underscore or hyphen), `kind` (`image`, `pdf`, `code`,
`video`, `audio`, or `text`), `mime`, a 64-character hexadecimal `sha256`, and
`path` relative to a configured artifact root. Inline data and unknown fields are
rejected.

Set `JEV_ARTIFACT_ROOTS` to an OS-path-separator-delimited list of absolute local
roots. The broker searches roots in order, rejects symlinks and traversal, reads
only regular files, and verifies SHA-256. It permits at most 8 references and
64 MiB per reference. Supported MIME values are explicitly listed in
`jev_laya_free.artifacts.MIMES`; MIME declarations are not content sniffing.

PDF references may select up to 4 distinct one-based `pages` (default `[1]`).
Image/PDF `crops` map page strings to at most 4 normalized `[x0,y0,x1,y1]` boxes;
images use key `"1"`, PDFs use selected page keys. Boxes must have positive area
within `[0,1]`. The PDF preprocessing adapter must check actual document page
counts before inference and raise `ArtifactSelectionError` for out-of-document
selections (returned as a safe 422); decoding and model preprocessing are not implemented
by the broker.

Artifact-capable backends implement `predict_artifacts(state, questions,
artifacts)`, receiving internal `ResolvedArtifact` objects with verified bytes
and no paths. Existing backends remain unchanged and return 503 for nonempty
artifact requests. Invalid references return 422; processing failures return
503 with no fallback decision. The deterministic workflow guard remains
unchanged and authoritative. Artifact response usage is restricted to the
standard token counts and broker-generated artifact ids, pages, and truncation
flags; backend usage extras are discarded. Raw bytes and filesystem paths are
never serialized by the broker or server.
Offline multimodal dataset preparation and source exclusions are documented in
[docs/DATASETS.md](docs/DATASETS.md). The converter records provenance, preserves
causal episode splits, and keeps local traces out of the default public build.
### Offline multimodal acceptance

`python -m jev_laya_free.trainer evaluate --predictions predictions.jsonl
--output acceptance.json --require-acceptance` emits the existing metrics plus an
`acceptance` report and exits 1 if any required gate fails. Without
`--require-acceptance`, the existing reporting exit behavior is preserved.
`--acceptance-config thresholds.json` accepts fields from
`training.acceptance.AcceptanceConfig`; the resolved configuration is included in
the report. This is a bounded offline capability check, never execution authority.
The deterministic guard remains authoritative; model outputs cannot authorize
writes, retries, or stops.

Inputs may be existing typed prediction records, with `question_name`, `labels`
(in probability order), `label_index`, `probabilities`, and `modality`, or paired
JSON rows such as:

```json
{"modality":"pdf","gold":{"loop_state":"repeat_without_progress","next_hand":"render_pdf_page"},"predictions":{"loop_state":"progress","next_hand":"extract_pdf_text"}}
```

Hermes `gold.loop_state` and normalized typed loop-state targets are authoritative
for repeat labels. Unrelated boolean questions are not repeat predictions.
Explicit legacy `expected_repeat`/`predicted_repeat` and
`expected_hand`/`predicted_hand` remain supported. High-risk code requires explicit
`gold.high_risk_code`/`predictions.high_risk_code` booleans (or
`expected_high_risk_code`/`predicted_high_risk_code`); review status alone does not
establish code risk. Modality can also come from metadata or state.

Defaults require guard precedence 100%, repeat and high-risk code recall .95 /
precision .90, next-hand macro-F1 .85 separately for code/PDF/image, PDF extract
versus render macro-F1 .90, image inspection recall .95, and top-label ECE <= .10
(ten equal-width bins). These are explicit initial policy thresholds, not measured
model claims. Each required metric needs at least `min_support=1`; binary recall
needs positive support. Missing evidence fails rather than passing vacuously.
Macro-F1 includes observed gold/predicted labels; the PDF gate always includes
both extract and render. Confusion matrices use gold rows and prediction columns.
Calibration requires typed probability records and reports pooled ECE, NLL, and
Brier score. Do not pool incompatible label spaces when interpreting its numeric
index confusion matrix.

Optional `latency_ms` and `vram_mb` samples are reported without a gate by default.
Set `max_latency_p95_ms` and `max_vram_mb` to require measured nearest-rank p95
latency and peak sampled VRAM. Measurements must be provided by the caller;
this harness does not benchmark hardware or import a model. Keep samples in
consistent units and avoid duplicating per-request measurements across questions.

## Shadow verification and preserved service boundaries

The Qwen backend was verified in an **isolated shadow process** (JEV-MM-17,
2026-09-21), not by replacing any live service. The exact commands (all
loopback-only; `<scratch>` is a writable scratch directory):

```bash
cd /home/coreys/models/jev-laya-free

# 1. Start the shadow (loopback 8094, distinct from the live 8093).
PYTHONPATH=src JEV_QWEN_BASE_URL=http://127.0.0.1:8080 JEV_QWEN_TIMEOUT=60 \
  JEV_ARTIFACT_ROOTS=<scratch>/artifacts \
  .venv/bin/python -m jev_laya_free.server --backend qwen --host 127.0.0.1 --port 8094

# 2. Confirm the bind is loopback-only (must be 127.0.0.1, not 0.0.0.0).
ss -ltnp | grep -E ':8094\b'

# 3. Run the live smoke (18/18) and the offline benchmark (no model/network).
#    (multimodal-smoke lives in the trainer module — a separate environment.)
python -m jev_laya_free.trainer multimodal-smoke --output-dir <scratch>/offline

# 4. Confirm the live services are unchanged (same PIDs/listeners).
ss -ltnp | grep -E ':8093\b|:8091\b'

# 5. Clean shutdown; 8094 is released, no lingering worker.
kill -TERM <shadow-pid>; sleep 2; ss -ltnp | grep -E ':8094\b'   # -> (empty)
```

**Preserved boundaries:** the live `127.0.0.1:8093` DistilBERT service and the
`127.0.0.1:8091` Laya sidecar were **not replaced** — same PIDs and listeners
before and after. The shadow ran on `127.0.0.1:8094` against the loopback Qwen
endpoint `http://127.0.0.1:8080`; no external network or media URL was used
(artifacts are local files sent as base64). Full evidence:
`reports/jev-mm17-shadow-verification.md`. No deployment is claimed beyond the
shadow in this release.

## Limitations and troubleshooting

- **Model serving was unavailable until verified.** The artifact broker, typed
  protocol, workflow guard, profile assessor, and offline benchmark existed
  before the Qwen backend; model serving for the multimodal modalities was not
  live-verified until JEV-MM-17. Do not treat the deterministic lexical backend
  as an intelligent safety classifier.
- **Unsupported modalities fail closed.** Unknown modalities and `audio` (the
  endpoint reports 0 audio slots, no transcription route) return a stable 503 /
  `audio_not_verified` — never a fallback to text. `pdf` is declared supported
  but its live probe verification is a future probe run; endpoint presence alone
  never counts as verified.
- **Strict score validation can fail closed live.** The endpoint is up and
  multimodal, but a non-conforming model answer (e.g. a `score` that does not
  match its own probability weighted-mean) fails closed (502/503, no fallback).
  The fake/injected tests are the deterministic contract; the live probe is
  reported as-is and does not substitute for them.
- **No model download or deployment.** The base package is standard-library
  only and does not download weights or run training; training lives in the
  trainer module in a separate environment. Checkpoints are installed via a
  checksummed manifest (`goblin-jev download-checkpoint`).
- **Troubleshooting:**
  - `503 backend unavailable` with no artifact roots set is expected (clean, no
    traceback, no filesystem/secret leakage). Set `JEV_ARTIFACT_ROOTS` to
    absolute roots.
  - `422` on artifact selection: the path escaped a root, the SHA-256 did not
    match, or a PDF page selection is out of document range.
  - `413` / `422` on the request: the body exceeded the bound or the question
    set exceeded the 33-question bound.
  - A non-conforming model output raises a stable code (`malformed_json`,
    `ValidationError`) — fail closed, no fallback decision.
  - `local-distilbert` import errors: install PyTorch/Transformers in a separate
    environment; the base package does not pull them.
