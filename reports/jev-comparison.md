Jev comparison: source-grounded protocol and semantic assessment

Date and provenance

Reference: https://github.com/kraayenjon/awesome-jev
Reference revision observed: 519a023a99c802dd7f75e0e7347a620f312670fd (latest commit shown by GitHub on 2026-09-20).
Local repository revision: 3ea96779c748560d1ec3d11f790be20daa3160d2.
Runtime: Python 3.11.15 on Linux.

Executive conclusion

A direct Jev-vs-local runtime benchmark could not be run without hosted Jev credentials or an approved external gateway. The referenced repository is an awesome-list, not the Jev implementation or model weights. This report therefore makes a protocol/semantic comparison and explicitly leaves Jev measurements unavailable. It does not substitute zeros or guessed Jev scores.

The local system and Jev have a comparable interface idea: typed questions whose answers are intended for software decisions rather than generated prose. Jev documents three primitives (Choice, Score, Noul), calibrated probabilities/confidence, and parallel evaluation. The local catalog uses the same three names and adds 12 application-oriented question families. The important architectural difference is that the local deterministic guard layer remains authoritative: advisory inference cannot authorize execution, and outages fail closed as errors.

Direct-runtime assessment

No local Jev implementation, weights, or runnable no-credential evaluator were found or downloaded. The reference documents `typesafe-sdk`, model alias `jev-latest` (documented current version `jev-1.13.0`), and endpoint `https://api.typesafe.ai/v1/systemone`; its quick start reads `TYPESAFE_API_KEY`. Using that path would require a hosted credential and was intentionally not attempted.

Unavailable measurements: Jev accuracy on the local fixture, Jev calibration (ECE/Brier/NLL), Jev latency/resource use on this host, and per-capability quality against the local labels. These are represented as null/unavailable in reports/jev-comparison.json.

Local raw protocol run

Command:
  .venv/bin/python scripts/benchmark.py --backend offline --fixture fixtures/benchmark-smoke.jsonl

Raw output: reports/jev-offline-smoke.json
Fixture SHA-256: 437200a626b1a6163ef94f9d72ec44c19e888e1131f7c3c2587fc7e3fc72573f
Result: 12 requests, 12 valid response-shaped rows, 0 errors, schema validity 1.0.

This is schema coverage only. The offline fixture contains placeholder prediction objects; it is not a learned-model evaluation and must not be read as local model accuracy.

Apples-to-apples boundary

The local held-out learned-model report is reports/final-capabilities.json. It evaluates 53 examples per capability and reports overall accuracy 0.2830188679245283, Brier 0.5901203496344464, ECE 0.2288995391637022, and NLL 0.9584843959493016. Those numbers are local DistilBERT measurements only, with post-hoc calibration, and cannot be compared to Jev without running Jev on identical labeled cases with identical scoring rules.

Semantic comparison

1. Typed answers
   Jev: Choice selects one option, Score rates a rubric, and Noul returns a 0–1 truth-like value; the reference says responses include probabilities/confidence where applicable.
   Local: taxonomy.py defines Choice, Score, and Noul for model_routing, confidence_action, tool_screening, progress, completion, skill_selection, compaction, citation, rag_filter, semantic_find, composite, and intent_routing.
   Assessment: interface-level alignment, not evidence of equivalent model quality.

2. Question coverage
   Jev is a general primitive for classify, route, score, detect, rank, extract, verify, and gate. The local system fixes 12 capability families for Hermes advisory workflows. Counts are not directly comparable because one is a general model API and the other is an application catalog.

3. Calibration
   The reference attributes calibrated probabilities to RLCD but this task did not reproduce a Jev calibration experiment. The local report provides held-out ECE/Brier/NLL and documents its post-hoc calibration caveat. No cross-model calibration claim is made.

4. Deterministic ownership and failure behavior
   The reference describes questions as inputs to application code, with code owning composition, thresholds, and side effects. The local implementation makes this operational: deterministic guards own precedence, advisory inference cannot authorize, and backend outage is recorded as an error. This is an architecture/protocol comparison, not a claim that Jev itself has failed-open behavior.

5. Latency and resources
   The reference README lists 70–500 ms end-to-end for Jev in its vendor-reported comparison, and says Jev is text-only with a 255-option Choice ceiling. These are source-listed claims, not measurements from this run. The local offline backend has no meaningful comparable inference latency; a local service run would measure transport/implementation latency, not Jev latency.

6. Per-capability quality
   Local per-capability learned metrics are in reports/final-capabilities.json. Jev per-capability metrics are unavailable because no Jev responses were obtained. No missing Jev value is represented as zero.

Reproduction and follow-up

The complete machine-readable comparison, including source claims, blocked measurements, local provenance, and null unavailable fields, is reports/jev-comparison.json. The raw local output is reports/jev-offline-smoke.json. A future credentialed run should use the same normalized states/questions, record the exact Jev model revision and SDK version, preserve raw responses, and score identical labels before making any learned-model comparison.

Sources

- awesome-jev README, What is Jev?, Jev vs LLM, Pricing/limits/access, Quick start: https://github.com/kraayenjon/awesome-jev/blob/519a023a99c802dd7f75e0e7347a620f312670fd/README.md
- Local taxonomy: src/jev_laya_free/taxonomy.py
- Local benchmark protocol: scripts/benchmark.py and docs/BENCHMARK.md
- Local learned evaluation: reports/final-capabilities.json
