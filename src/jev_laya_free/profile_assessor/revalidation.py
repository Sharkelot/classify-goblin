"""Dispatcher revalidation and fallback chain (JEV-MM-11).

The assessor is advisory only. Before the dispatcher assigns, it calls
``revalidate`` with a fresh roster snapshot and the entry digest captured
at assessment time. A mismatch means the profile changed (or vanished)
between assessment and assignment, so the dispatcher falls back to the
next candidate via ``first_valid_candidate``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RevalidationResult:
    status: str  # 'ok' | 'stale' | 'missing' | 'invalid'


def revalidate(name, fresh_snapshot, expected_digest):
    """Revalidate one candidate against a fresh roster snapshot.

    Returns a RevalidationResult:
      - 'missing':  the profile is not in the fresh roster
      - 'invalid':  the profile is not in the fresh valid_assignees set
      - 'stale':    the entry digest changed (profile was modified)
      - 'ok':       everything matches
    """
    entry = None
    for e in fresh_snapshot.entries:
        if e.name == name:
            entry = e
            break
    if entry is None:
        return RevalidationResult('missing')
    if name not in fresh_snapshot.valid_assignees:
        return RevalidationResult('invalid')
    if entry.entry_digest() != expected_digest:
        return RevalidationResult('stale')
    return RevalidationResult('ok')


def first_valid_candidate(candidates, fresh_snapshot):
    """Walk the ranked candidate list; return the first profile that
    revalidates to 'ok', or None if none do."""
    for c in candidates:
        name = c['profile']
        # Find the original entry digest from the candidate's stored digest.
        # We need the expected digest — it was captured at assessment time.
        # The candidate dict carries 'entry_digest' for this purpose.
        expected = c.get('entry_digest')
        if expected is None:
            continue
        rv = revalidate(name, fresh_snapshot, expected)
        if rv.status == 'ok':
            return name
    return None
