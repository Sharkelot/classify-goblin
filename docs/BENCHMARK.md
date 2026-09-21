# Reproducible benchmark

The benchmark is dependency-light and uses the same normalized request schema for the 12 capability families and the legacy workflow questions (`next_hand` and `needs_review`). It never treats a rules response as a learned-model result.

## Smoke test (offline, no service or hosted API)

```bash
PYTHONPATH=src python scripts/benchmark.py --backend offline --fixture fixtures/benchmark-smoke.jsonl
```

The fixture is committed and its SHA-256, git commit, Python version, seed, checkpoint, and service URL are recorded in JSON output. Add `--markdown` for a readable summary.

## Rules backend

In one terminal, start the loopback-only service:

```bash
PYTHONPATH=src python -m jev_laya_free.server --backend rules --port 8093
```

Then run:

```bash
PYTHONPATH=src python scripts/benchmark.py --backend rules --url http://127.0.0.1:8093
```

Use `--backend laya --checkpoint /path/to/checkpoint` only with an already-running compatible local service. No model download or deployment is performed by this command. The client rejects non-loopback URLs.

## Full held-out evaluation

The full fixture is generated deterministically by the training benchmark builder. The `capability-benchmark` command has no `--full` flag; the full fixture is selected with `--variants 76` (the default `--variants 5` produces the 420-example smoke fixture). After preparing it, run:

```bash
python -m jev_laya_free.trainer capability-benchmark --variants 76 --output-dir data/capability-benchmark-full
PYTHONPATH=src python scripts/benchmark.py --backend rules --fixture data/capability-benchmark-full/capability-benchmark-test.jsonl
```

The full fixture is 6384 examples: 4476 train, 1272 validation, and a 636-example held-out test split (53 per capability). The 1908 figure is the combined validation+test support (159 per capability), not the test split size.

For an offline saved evaluator, provide JSONL rows with a `prediction` (or `predictions`) object containing every catalog question name and use `--backend offline`. Offline results report schema coverage only; learned-model quality must be reported separately using the model evaluation scripts and must label raw versus post-hoc test calibration. The capability checkpoint's temperature was fitted on the held-out test split itself, so its held-out numbers are post-hoc test calibration; label them as such rather than as raw held-out quality. Rules results report deterministic behavior and transport/schema outcomes, not Jev-model accuracy.

The report includes p50/p95 request latency, per-capability counts, deterministic guard precedence, fail-open semantics, and exact environment metadata. A backend outage is recorded as an error; the advisory layer never authorizes execution when inference is unavailable.
