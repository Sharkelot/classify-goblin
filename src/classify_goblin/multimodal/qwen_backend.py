"""Qwen backend: manifest-gated, fail-closed, no raw bytes in the record.

The backend consults the versioned modality manifest (local probe
observations only) before accepting a payload. Video additionally requires an
explicit probe-verified flag. Unknown modalities, empty payloads, and
unverified modalities all fail closed with a stable reason.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from . import probe as _probe
from .media import prepare_media
from .preprocessing import DEFAULT_LIMITS


@dataclass(frozen=True)
class BackendResult:
    modality: str
    status: str
    reason: str = ''
    sha256: str = ''
    features: dict = field(default_factory=dict)
    data: bytes = field(default=b'', repr=False)

    def usage(self):
        return {'modality': self.modality, 'status': self.status,
                'reason': self.reason, 'sha256': self.sha256}


class QwenBackend:
    """Fail-closed adapter for the local Qwen endpoint (CG-MM-01 probe)."""

    def __init__(self, manifest_path=None, local_capabilities=None, limits=None):
        self._manifest = _probe.load_manifest(manifest_path) if manifest_path else _probe.default_manifest()
        self._local_capabilities = local_capabilities or {}
        self._limits = limits or DEFAULT_LIMITS

    def _supported(self, modality):
        if self._local_capabilities:
            return bool(self._local_capabilities.get(modality))
        return _probe.is_available(self._manifest, modality)

    def run(self, modality, data, *, probe_verified=False):
        if not isinstance(data, bytes) or not data:
            return BackendResult(modality, 'unavailable', 'empty_payload')
        if not self._supported(modality):
            return BackendResult(modality, 'unavailable', 'modality_not_supported')
        if modality == 'video' and not probe_verified:
            return BackendResult(modality, 'unavailable', 'video_not_verified')
        digest = hashlib.sha256(data).hexdigest()
        kind = {'image': 'image', 'text': 'text', 'code': 'code',
                'video': 'image', 'pdf': 'pdf'}.get(modality, 'image')
        from ..artifacts import ResolvedArtifact
        artifact = ResolvedArtifact('backend', kind, 'application/octet-stream',
                                   digest, (), {}, data)
        payload = prepare_media(artifact, modality=modality,
                               probe_verified=probe_verified, limits=self._limits)
        if payload.status == 'unavailable':
            return BackendResult(modality, 'unavailable', payload.reason,
                                sha256=digest)
        return BackendResult(modality, 'ok', sha256=digest,
                             features=payload.features, data=payload.data)
