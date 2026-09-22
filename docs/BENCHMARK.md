# Reproducible benchmark

The benchmark is dependency-light and uses the same normalized request schema for the 12 capability families and the legacy workflow questions (`next_hand` and `needs_review`). It never treats a rules response as a learned-model result.

## Smoke test (offline, no service or hosted API)

```bash
PYTHONPATH=src python scripts/benchmark.py --backend offline --fixture fixtures/benchmark-smoke.jsonl
```

The fixture is committed and its SHA-256, git commit, Python version, seed, checkpoint, and service URL are recorded in JSON output. Add `--markdown` for a readable summary. The committed result is `reports/classify-goblin-offline-smoke.json`.

## Rules backend

In one terminal, start the loopback-only service:

```bash
PYTHONPATH=src python -m classify_goblin.server --backend rules --port 8093
```

Then run:

```bash
PYTHONPATH=src python scripts/benchmark.py --backend rules --url http://127.0.0.1:8093
```

Use `--backend laya --checkpoint /path/to/checkpoint` only with an already-running compatible local service. No model download or deployment is performed by this command. The client rejects non-loopback URLs.

## What the benchmark reports

Offline results report schema coverage only. Rules results report deterministic
behavior and transport/schema outcomes, not learned-model accuracy. The report
includes p50/p95 request latency, per-capability counts, deterministic guard
precedence, fail-open semantics, and exact environment metadata. A backend
outage is recorded as an error; the advisory layer never authorizes execution
when inference is unavailable.

## Learned-model quality is out of scope

Training, dataset generation, and the full held-out evaluation are not part of
this runtime-only release; the `classify_goblin.trainer` module does not exist in
this repository. A learned-model quality claim requires held-out support from
the separate training environment and must label raw versus post-hoc
calibration. This benchmark never produces one.
