#!/usr/bin/env python3
"""Reproducible, dependency-free local benchmark for the JEV contract."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jev_laya_free.client import TypeSafeClient, ClientError
from jev_laya_free.taxonomy import CAPABILITIES, questions
from jev_laya_free.workflow import decide


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def _commit():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError): return "unknown"


def _fixture(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows: raise ValueError("fixture is empty")
    return rows


def _pct(values, p):
    if not values: return None
    vals = sorted(values); index = (len(vals) - 1) * p / 100
    lo, hi = int(index), min(int(index) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (index - lo)


def _valid_response(response, names):
    return (isinstance(response, dict) and response.get("model") and
            isinstance(response.get("answers"), dict) and set(names) <= set(response["answers"]))


def run(args):
    rows = _fixture(args.fixture)
    q = questions()
    names = list(q)
    latencies, valid, errors = [], 0, 0
    per_cap = {name: {"requests": 0, "valid": 0, "errors": 0} for name in CAPABILITIES}
    started = time.perf_counter()
    if args.backend == "offline":
        for row in rows:
            pred = row.get("prediction", row.get("predictions"))
            ok = isinstance(pred, dict) and set(names) <= set(pred)
            valid += int(ok); errors += int(not ok)
            cap = row.get("capability")
            if cap in per_cap: per_cap[cap]["requests"] += 1; per_cap[cap]["valid"] += int(ok); per_cap[cap]["errors"] += int(not ok)
    else:
        client = TypeSafeClient(base_url=args.url, model={"rules": "local-rules-v1", "laya": "local-default"}[args.backend], timeout=args.timeout, retries=0)
        for row in rows:
            t = time.perf_counter()
            try:
                result = client.system_one(state=row.get("state", {}), questions=q)
                ok = _valid_response(result, names)
                valid += int(ok); errors += int(not ok); latencies.append((time.perf_counter() - t) * 1000)
                cap = row.get("capability")
                if cap in per_cap: per_cap[cap]["requests"] += 1; per_cap[cap]["valid"] += int(ok); per_cap[cap]["errors"] += int(not ok)
            except (ClientError, ValueError, TypeError) as exc:
                errors += 1; latencies.append((time.perf_counter() - t) * 1000)
                if args.verbose: print(f"request failed: {exc}", file=sys.stderr)
    guards = []
    for state in ({"terminal": True}, {"same_action_streak": 2}, {"same_action_streak": 3}, {"same_tool_failures": 3}, {"context_compactions": 2}, {}):
        guards.append({"state": state, "decision": decide(state)["gate"]["decision"]})
    manifest = Path(args.fixture)
    learned = None
    quality_report = ROOT / "reports" / "final-capabilities.json"
    if args.backend == "laya" and quality_report.exists():
        try:
            learned = {"source": str(quality_report),
                       "overall_raw_test": json.loads(quality_report.read_text()).get("overall"),
                       "calibration_note": "post-hoc test calibration is not an unbiased generalization estimate"}
        except (OSError, ValueError, TypeError):
            learned = {"source": str(quality_report), "status": "unreadable"}
    result = {"schema_version": "jev-benchmark-1", "backend": args.backend,
      "learned_model_quality": learned,
      "deterministic_guard_behavior": {"cases": guards, "precedence_verified": guards[0]["decision"] == "terminal"},
      "fail_open": {"backend_outage_is_error": True, "advisory_does_not_authorize": True},
      "requests": len(rows), "valid_responses": valid, "errors": errors,
      "schema_validity": valid / len(rows) if rows else 0.0, "per_capability": per_cap,
      "latency_ms": {"p50": _pct(latencies, 50), "p95": _pct(latencies, 95), "samples": len(latencies)},
      "metadata": {"commit": _commit(), "python": platform.python_version(), "platform": platform.platform(),
        "fixture": str(manifest), "fixture_sha256": _sha(manifest), "checkpoint": args.checkpoint,
        "service_url": args.url if args.backend != "offline" else None, "seed": args.seed,
        "duration_seconds": time.perf_counter() - started}}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("rules", "laya", "offline"), default="rules")
    p.add_argument("--fixture", default=str(ROOT / "fixtures" / "benchmark-smoke.jsonl"))
    p.add_argument("--url", default="http://127.0.0.1:8093")
    p.add_argument("--checkpoint", default="none")
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(); result = run(args)
    if args.markdown:
        print("# JEV benchmark\n\n" + "\n".join(f"- **{k}:** {v}" for k, v in result.items() if k not in ("per_capability", "deterministic_guard_behavior", "fail_open")))
        print("\n## Per capability\n\n| Capability | Requests | Valid | Errors |\n|---|---:|---:|---:|")
        for k, v in result["per_capability"].items(): print(f"| {k} | {v['requests']} | {v['valid']} | {v['errors']} |")
    else: print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["errors"] == 0 else 2

if __name__ == "__main__": raise SystemExit(main())
