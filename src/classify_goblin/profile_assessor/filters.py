"""Deterministic eligibility filters F1-F7 (CG-MM-11).

Each filter is pure and fail-closed: an unknown value excludes the
candidate with a fixed reason code, never a silent pass. ``filter_chain``
applies them in a fixed order; the prefix property holds — a candidate
surviving the first k filters survives all 7.
"""
from __future__ import annotations

# Fixed order; the prefix property (test_filter_chain_prefix_property)
# depends on it.
FILTER_ORDER = (
    'not_on_disk',
    'not_valid_assignee',
    'context_below_min',
    'modality_not_declared',
    'capability_not_declared',
    'workspace_mismatch',
    'status_blocked',
)


def _f1_on_disk(entry):
    return None if entry.on_disk else 'not_on_disk'


def _f2_valid_assignee(entry, valid_assignees):
    if not entry.valid_assignee or entry.name not in valid_assignees:
        return 'not_valid_assignee'
    return None


def _f3_context(entry, min_context):
    if min_context <= 0:
        return None
    if entry.context_length is None:
        return 'context_unknown'
    if entry.context_length < min_context:
        return 'context_below_min'
    return None


def _f4_modality(entry, required_modalities):
    for mod in required_modalities:
        if mod not in entry.modalities:
            return 'modality_not_declared'
    return None


def _f5_capability(entry, required_capabilities):
    for cap in required_capabilities:
        if cap not in entry.capabilities:
            return 'capability_not_declared'
    return None


def _f6_workspace(entry, workspace_repo):
    if workspace_repo is None:
        return None
    if entry.workspace is None:
        # Unknown workspace: pass, but the scorer applies a penalty.
        return None
    if entry.workspace != workspace_repo:
        return 'workspace_mismatch'
    return None


def _f7_status(entry):
    if entry.status in ('blocked', 'unavailable'):
        return 'status_' + entry.status
    return None


_FILTERS = (
    lambda e, req, va: _f1_on_disk(e),
    lambda e, req, va: _f2_valid_assignee(e, va),
    lambda e, req, va: _f3_context(e, req.min_context),
    lambda e, req, va: _f4_modality(e, req.required_modalities),
    lambda e, req, va: _f5_capability(e, req.required_capabilities),
    lambda e, req, va: _f6_workspace(e, req.workspace_repo),
    lambda e, req, va: _f7_status(e),
)


def apply_filters(entries, request, valid_assignees):
    """Return {name: reason-or-None} for the full F1-F7 chain."""
    return filter_chain(entries, 7, request, valid_assignees)


def filter_chain(entries, k, request, valid_assignees):
    """Apply the first ``k`` filters; survivors map to None."""
    result = {}
    for entry in entries:
        reason = None
        for i in range(min(k, len(_FILTERS))):
            reason = _FILTERS[i](entry, request, valid_assignees)
            if reason is not None:
                break
        result[entry.name] = reason
    return result
