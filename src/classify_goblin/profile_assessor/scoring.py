"""Advisory fit scoring and verdict (CG-MM-11).

The score is a transparent, deterministic heuristic — NOT a learned model.
It is advisory only: the dispatcher revalidates and owns every side effect.

Fit components (sum, clamped to [0, 1]):
  - capability coverage: 0.40 * (matched / required)   (1.0 if none required)
  - modality coverage:   0.20 * (matched / required)   (1.0 if none required)
  - context headroom:    0.15 * min(1, ctx / min_ctx) (1.0 if no min)
  - description present: 0.10 (0.00 if null)
  - skill coverage:      0.15 (1.0 if any skills, 0.50 if none)

Confidence starts at 1.0 and is reduced by fixed penalties:
  - description_auto:        -0.25
  - status unknown/blocked:  -0.15
  - workspace unknown:       -0.15
  - context unknown:         -0.10
  - snapshot age > 600s:     -0.10
Clamped to [0, 1].

``now`` is injectable so tests are deterministic; it defaults to the
current UTC time.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Verdict thresholds (fixed, from the design contract).
AMBIGUITY_EPSILON = 0.06
LOW_CONFIDENCE = 0.40
MAX_SNAPSHOT_AGE_S = 300


def _parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None


def _age_seconds(snapshot_time, now):
    """Return the snapshot age in seconds, or None if unknown.

    When ``now`` is None the age is 0.0 (deterministic: treat as fresh).
    Pass an explicit ``now`` to test staleness.
    """
    snap_ts = _parse_ts(snapshot_time)
    if snap_ts is None:
        return None
    if now is None:
        return 0.0
    return (now - snap_ts).total_seconds()


def score_candidate(entry, request, snapshot_time, now=None):
    """Return (fit, confidence, reasons) for a single eligible entry."""
    reasons = []

    # --- fit ---
    if request.required_capabilities:
        matched = sum(1 for c in request.required_capabilities
                      if c in entry.capabilities)
        cap_score = matched / len(request.required_capabilities)
    else:
        cap_score = 1.0
    if request.required_modalities:
        matched = sum(1 for m in request.required_modalities
                      if m in entry.modalities)
        mod_score = matched / len(request.required_modalities)
    else:
        mod_score = 1.0
    if request.min_context > 0:
        if entry.context_length is None:
            ctx_score = 0.0
        else:
            ctx_score = min(1.0, entry.context_length / request.min_context)
    else:
        ctx_score = 1.0
    desc_score = 1.0 if entry.description else 0.0
    skill_score = 1.0 if entry.skill_names else 0.5

    fit = (0.40 * cap_score + 0.20 * mod_score + 0.15 * ctx_score
           + 0.10 * desc_score + 0.15 * skill_score)
    fit = max(0.0, min(1.0, fit))

    # --- confidence ---
    conf = 1.0
    if entry.description_auto:
        conf -= 0.25
        reasons.append('description_auto')
    if entry.status in ('unknown', 'blocked', 'unavailable'):
        conf -= 0.15
        reasons.append('status_' + str(entry.status))
    if request.workspace_repo is not None and entry.workspace is None:
        conf -= 0.15
        reasons.append('compat_unknown')
    if entry.context_length is None and request.min_context > 0:
        conf -= 0.10
        reasons.append('context_unknown')
    age = _age_seconds(snapshot_time, now)
    if age is not None and age > 600:
        conf -= 0.10
        reasons.append('snapshot_stale')
    conf = max(0.0, min(1.0, conf))

    return fit, conf, reasons


def decide_verdict(candidates, request, snapshot_time, now=None):
    """Return (verdict, reason) per the contract's decision rules.

    Order (from the design contract): no-eligible, high-risk, ambiguity,
    low-confidence, stale-snapshot, otherwise recommend.
    """
    if not candidates:
        return 'abstain', 'no_eligible_candidate'
    if request.risk == 'high':
        return 'review', 'high_risk_task'
    if len(candidates) >= 2:
        top = candidates[0]
        second = candidates[1]
        if top['confidence'] - second['confidence'] < AMBIGUITY_EPSILON:
            return 'review', 'ambiguous_candidates'
    if candidates[0]['confidence'] < LOW_CONFIDENCE:
        return 'review', 'low_confidence'
    age = _age_seconds(snapshot_time, now)
    if age is not None and age > MAX_SNAPSHOT_AGE_S:
        return 'review', 'stale_snapshot'
    return 'recommend', None
