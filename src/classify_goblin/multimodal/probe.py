"""Capability probe and versioned modality manifest (V1).

Local observations only — upstream claims never count as local support. A
modality is ``verified`` only when the local probe (CG-MM-01) exercised it
against the live endpoint and observed a successful decode. ``unavailable``
means the probe observed a hard rejection (400/404). ``not_verified`` means no
local probe ran for that modality.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

# SHA256 of the CG-MM-01 runtime probe report (local observation evidence).
PROBE_REPORT_SHA256 = '31ed208a11d720d7aa06f0c87b74923318ed4f4cc41f4ac9a3cb6005e7997e3e'

# Local observations from the CG-MM-01 probe (not upstream claims).
_LOCAL_OBSERVATIONS = {
    'text': 'verified',
    'image': 'verified',
    'video': 'verified',
    'audio': 'unavailable',
    'pdf': 'unavailable',
}


def default_manifest():
    """Build the versioned modality manifest from local probe observations."""
    capabilities = {}
    for name, state in _LOCAL_OBSERVATIONS.items():
        capabilities[name] = {'state': state, 'source': 'local_probe',
                              'evidence': 'CG-MM-01'}
    return {
        'version': 1,
        'probed_at': datetime.now(timezone.utc).isoformat(),
        'capabilities': capabilities,
        'evidence': {'probe_report_sha256': PROBE_REPORT_SHA256},
    }


def is_available(manifest, modality):
    """A modality is available only when its local probe state is verified."""
    state = manifest['capabilities'].get(modality, {}).get('state')
    return state == 'verified'


def load_manifest(path):
    """Load a manifest from disk; fall back to the default on any error."""
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        if data.get('version') != 1 or 'capabilities' not in data:
            raise ValueError('invalid manifest version')
        return data
    except (OSError, ValueError):
        return default_manifest()
