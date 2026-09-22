"""Hermes profile assessor (CG-MM-11).

Advisory-only profile fit assessment for Kanban task routing.

- Deterministic filters F1-F7 (fail-closed) decide eligibility.
- A transparent heuristic scores fit/confidence for eligible profiles.
- The verdict (recommend / review / abstain) is advisory: the dispatcher
  revalidates the selected profile and owns every side effect.

No model output ever authorizes execution or delegation.
"""
from __future__ import annotations

import hashlib
import json

from .schema import AssessmentRequest
from .roster import (
    RosterSnapshot,
    ProfileEntry,
    parse_profile_dir,
    parse_profiles_root,
)
from .filters import apply_filters
from .scoring import score_candidate, decide_verdict
from .revalidation import revalidate, first_valid_candidate

__all__ = [
    'AssessmentRequest',
    'RosterSnapshot',
    'ProfileEntry',
    'parse_profile_dir',
    'parse_profiles_root',
    'revalidate',
    'first_valid_candidate',
    'assess',
]


def _digest_of(obj):
    payload = json.dumps(obj, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def assess(request, snapshot, now=None):
    """Assess every roster entry against the request.

    Returns a JSON-serializable dict:
      verdict, reason, candidates (ranked, with fit/confidence/evidence),
      excluded (with fixed reason codes), advisory_only, digests.
    """
    filters = apply_filters(snapshot.entries, request,
                           snapshot.valid_assignees)

    candidates = []
    excluded = []
    for entry in snapshot.entries:
        reason = filters.get(entry.name)
        if reason is not None:
            excluded.append({'profile': entry.name, 'filter_reason': reason})
            continue
        fit, conf, evidence = score_candidate(entry, request,
                                            snapshot.snapshot_time, now=now)
        candidates.append({
            'profile': entry.name,
            'eligible': True,
            'fit': round(fit, 4),
            'confidence': round(conf, 4),
            'evidence': evidence,
            'entry_digest': entry.entry_digest(),
        })

    # Rank: confidence desc, then fit desc, then name asc (deterministic).
    candidates.sort(key=lambda c: (-c['confidence'], -c['fit'], c['profile']))
    for i, c in enumerate(candidates):
        c['rank'] = i + 1

    verdict, reason = decide_verdict(candidates, request,
                                    snapshot.snapshot_time, now=now)

    return {
        'verdict': verdict,
        'reason': reason,
        'candidates': candidates,
        'excluded': excluded,
        'advisory_only': True,
        'roster_digest': snapshot.digest,
        'request_digest': _digest_of(request.to_dict()),
    }
