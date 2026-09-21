"""JEV-MM-13: bounded audio adapter — fail-closed, no fake success metrics.

The local Qwen endpoint rejects all audio parts (limit 0) and has no
transcription route (JEV-MM-13 live probe, 2026-09-21). Audio is therefore
fail-closed: every path returns an explicit, stable ``unavailable`` result
with a documented reason. The adapter boundary (sample-rate, duration,
channel, byte, and chunk limits) is still enforced so a future verified
STT/omni runtime can plug in without a contract change.
"""
import hashlib
import struct
import unittest
import wave
import io

from jev_laya_free.artifacts import ResolvedArtifact
from jev_laya_free.multimodal import prepare_audio
from jev_laya_free.multimodal.audio import AudioLimits


def _wav_bytes(seconds=0.1, rate=8000, channels=1):
    """Synthetic silence WAV; no private media."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(channels)
        w.setsampwidth(1)
        w.setframerate(rate)
        w.writeframes(b'\x00' * (int(rate * seconds) * channels))
    return buf.getvalue()


def _artifact(data, mime='audio/wav'):
    return ResolvedArtifact('a1', 'audio', mime,
                           hashlib.sha256(data).hexdigest(), (), {}, data)


class AudioFailClosedTests(unittest.TestCase):
    def test_unavailable_by_default(self):
        payload = prepare_audio(_artifact(_wav_bytes()))
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'audio_not_verified')

    def test_unavailable_with_explicit_no_capability(self):
        payload = prepare_audio(_artifact(_wav_bytes()), capability_verified=False)
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'audio_not_verified')

    def test_empty_payload(self):
        payload = prepare_audio(_artifact(b''))
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'empty_payload')

    def test_oversize_bytes(self):
        payload = prepare_audio(_artifact(b'\x00' * (9 * 1024 * 1024)),
                                limits=AudioLimits(max_bytes=8 * 1024 * 1024))
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'invalid_or_oversize_bytes')

    def test_oversized_duration(self):
        # 31 s of silence > 30 s limit.
        payload = prepare_audio(_artifact(_wav_bytes(seconds=31)))
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'audio_too_long')

    def test_sample_rate_too_high(self):
        payload = prepare_audio(_artifact(_wav_bytes(seconds=0.1, rate=96000)))
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'sample_rate_too_high')

    def test_too_many_channels(self):
        payload = prepare_audio(_artifact(_wav_bytes(channels=3)))
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'too_many_channels')

    def test_unsupported_format(self):
        # FLAC magic (fLaC) is not in the allow-list.
        payload = prepare_audio(_artifact(b'fLaC' + b'\x00' * 100, mime='audio/flac'))
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'unsupported_audio_format')

    def test_malformed_container(self):
        # A RIFF header that is not a valid WAV.
        payload = prepare_audio(_artifact(b'RIFF' + b'\x00' * 100))
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'audio_decode_failed')

    def test_raw_media_never_in_repr(self):
        data = _wav_bytes() + b'Z' * 500
        payload = prepare_audio(_artifact(data))
        self.assertNotIn('Z' * 100, repr(payload))
        self.assertEqual(payload.data, b'')

    def test_no_transcript_in_metadata(self):
        # The record must not carry any transcript or raw audio content.
        payload = prepare_audio(_artifact(_wav_bytes()))
        for key in payload.features:
            self.assertNotIn('transcript', key.lower())
        self.assertEqual(payload.features.get('sample_rate'), 8000)
        self.assertEqual(payload.features.get('channels'), 1)


class AudioLimitsTests(unittest.TestCase):
    def test_limits_validation(self):
        with self.assertRaises(ValueError):
            AudioLimits(max_bytes=0)
        with self.assertRaises(ValueError):
            AudioLimits(max_duration_seconds=0)
        with self.assertRaises(ValueError):
            AudioLimits(max_channels=0)
        with self.assertRaises(ValueError):
            AudioLimits(max_sample_rate_hz=0)
        with self.assertRaises(ValueError):
            AudioLimits(max_chunk_bytes=0)


if __name__ == '__main__':
    unittest.main()
