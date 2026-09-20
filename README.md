# jev-laya-free

An independent, MIT-licensed local implementation of the documented Jev typed-decision
HTTP shape, with a Python client and optional Laya inference. This is **not TypeSafe's
hosted model**, an official TypeSafe SDK, a reproduction of Jev's weights, or a promise
of equivalent predictions. Local confidence is **not calibrated Jev confidence**.

No hosted API, model weights, private traces, or secrets are included. The default backend
is a deterministic lexical demo for testing. It does not understand arbitrary instructions
and must not be used as an intelligent safety classifier. The optional Laya backend is
advisory; its reliability must be evaluated on your own data. “Free” refers to this source
and local operation without a hosted API fee; model licensing and hardware costs are separate.

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

`POST /v1/systemone` accepts `state`, `questions`, and optional `model`. State may be a string,
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
`jev_laya_free`, not an import-compatible replacement for `typesafe_sdk`.

The client accepts only loopback HTTP origins, ignores proxy environment settings, refuses
redirects, and validates response types, names, distributions, scores and usage. It raises
`ClientError` on failed calls; it never fabricates a successful fallback decision. Invalid
local requests raise `ValidationError`. Timeouts default to 5 seconds per socket operation;
retries default to 2, with bounded exponential backoff for transport errors and HTTP
429/502/503/504/529. Validation and auth failures are not retried. The maximum is 5 retries.
A timeout does not cancel backend inference and is not a whole-call wall-clock deadline.

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
No hosted requests or official SDK interoperability tests were performed.

The canonical reference specifies Choice `criteria`; `options` is an intentional local alias
for callers needing that shape. The official API marks `model` required; this service makes it
optional with `local-default`. `request_id` is a local extension, absent from the inspected
response schema. The reference accepts structured Score criteria but describes legend values
as strings; compact JSON serialization is our explicit resolution. The official minimum Choice
cardinality and maximum question count are not fully specified there; local bounds above are
implementation policy. Confidence calibration/formula, performance, context capacity, prediction
quality, error body details, model discovery, streaming, async clients, hosted authentication,
and all official SDK behavior are outside the compatibility claim.

## Offline checks

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests examples
PYTHONPATH=src python3 examples/workflow.py
```

Tests create ephemeral loopback servers and use synthetic data, the rules backend, or mocks.
They exercise schemas, typed roundtrips, auth, failures, retries, Laya translation, and deterministic
safety precedence. No private traces, real model weights, hosted calls, or running gateways are needed.
