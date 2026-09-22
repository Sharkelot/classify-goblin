"""Bounded audio adapter: fail-closed, no fake success metrics.

The local Qwen endpoint rejects all audio parts (limit 0) and has no
transcription route (CG-MM-13 live probe, 2026-09-21). Audio is therefore
fail-closed: every path returns an explicit, stable ``unavailable`` result
with a documented reason. The adapter boundary (sample-rate, duration,
channel, byte, and chunk limits) is still enforced so a future verified
STT/omni runtime can plug in without a contract change.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from ..artifacts import ResolvedArtifact
from .media import MediaPayload, _bad

# Allowed audio MIME types (container allow-list).
_AUDIO_MIME = {'audio/wav', 'audio/x-wav', 'audio/mpeg', 'audio/mp3'}


@dataclass(frozen=True)
class AudioLimits:
    max_bytes: int = 8 * 1024 * 1024
    max_duration_seconds: int = 30
    max_channels: int = 2
    max_sample_rate_hz: int = 48000
    max_chunk_bytes: int = 1 * 1024 * 1024

    def __post_init__(self):
        for name in ('max_bytes', 'max_duration_seconds', 'max_channels',
                     'max_sample_rate_hz', 'max_chunk_bytes'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')


DEFAULT_AUDIO_LIMITS = AudioLimits()


def _parse_wav_header(data: bytes):
    """Return (sample_rate, channels, duration) from a RIFF/WAV header, or None."""
    if len(data) < 44 or data[:4] != b'RIFF' or data[8:12] != b'WAVE':
        return None
    # Walk the RIFF chunks to find 'fmt '.
    pos = 12
    sample_rate = 0
    channels = 0
    bits_per_sample = 8
    fmt_found = False
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        try:
            chunk_size = struct.unpack('<I', data[pos + 4:pos + 8])[0]
        except struct.error:
            return None
        if chunk_id == b'fmt ':
            if pos + 8 + 16 > len(data):
                return None
            # fmt chunk layout: audio_format(H) channels(H) sample_rate(I)
            # byte_rate(I) block_align(H) bits_per_sample(H)
            audio_format, channels = struct.unpack(
                '<HH', data[pos + 8:pos + 8 + 4])
            sample_rate = struct.unpack(
                '<I', data[pos + 8 + 4:pos + 8 + 8])[0]
            # bits_per_sample is the last 2 bytes of the 16-byte fmt payload.
            bits_per_sample = struct.unpack(
                '<H', data[pos + 8 + 14:pos + 8 + 16])[0]
            if audio_format != 1:  # PCM only
                return None
            fmt_found = True
            break
        pos += 8 + chunk_size + (chunk_size % 2)
    if not fmt_found:
        return None
    # Walk to 'data' chunk for duration.
    pos = 12
    data_size = 0
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        try:
            chunk_size = struct.unpack('<I', data[pos + 4:pos + 8])[0]
        except struct.error:
            break
        if chunk_id == b'data':
            data_size = chunk_size
            break
        pos += 8 + chunk_size + (chunk_size % 2)
    # Duration = data_size / (sample_rate * channels * bytes_per_sample).
    bytes_per_sample = bits_per_sample // 8
    byte_rate = sample_rate * channels * bytes_per_sample
    duration = data_size / byte_rate if byte_rate > 0 else 0.0
    return sample_rate, channels, duration


def prepare_audio(artifact, *, capability_verified=False,
                  limits=DEFAULT_AUDIO_LIMITS):
    """Fail closed unless a verified STT/omni runtime is available."""
    data = artifact.data
    if not isinstance(data, bytes) or not data:
        return _bad('audio', 'audio', artifact.mime, 'empty_payload')
    if len(data) > limits.max_bytes:
        return _bad('audio', 'audio', artifact.mime, 'invalid_or_oversize_bytes')
    # Container MIME allow-list check.
    if artifact.mime not in _AUDIO_MIME:
        return _bad('audio', 'audio', artifact.mime, 'unsupported_audio_format')
    # Parse WAV header for sample rate, channels, duration.
    wav_info = _parse_wav_header(data)
    if wav_info is None:
        return _bad('audio', 'audio', artifact.mime, 'audio_decode_failed')
    sample_rate, channels, duration = wav_info
    # Enforce sample-rate, channel, and duration limits.
    if sample_rate > limits.max_sample_rate_hz:
        return _bad('audio', 'audio', artifact.mime, 'sample_rate_too_high')
    if channels > limits.max_channels:
        return _bad('audio', 'audio', artifact.mime, 'too_many_channels')
    if duration > limits.max_duration_seconds:
        return _bad('audio', 'audio', artifact.mime, 'audio_too_long')
    # Fail-closed: no verified capability.
    if not capability_verified:
        return MediaPayload('audio', 'audio', artifact.mime, 'unavailable',
                           'audio_not_verified',
                           features={'sample_rate': sample_rate,
                                    'channels': channels,
                                    'duration': round(duration, 3)},
                           data=b'')
    # A verified runtime would produce a transcript here; for now, ok with no transcript.
    return MediaPayload('audio', 'audio', artifact.mime, 'ok', '',
                       features={'sample_rate': sample_rate,
                                'channels': channels,
                                'duration': round(duration, 3)},
                       capabilities={'transcription': 'available'},
                       data=b'')
