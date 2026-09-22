"""CG-MM-13: bounded video adapter — injected decoder tests + real ffmpeg/ffprobe.

Covers: fail-closed without probe, verified no-sampler, sampler ok, sampler
failure (partial), oversized bytes, oversized duration, frame-budget overflow,
timestamp ordering, parser timeout, unsupported runtime, malformed container,
raw-media non-leakage, and the default ffprobe/ffmpeg frame sampler.
"""
import hashlib
import struct
import unittest

from classify_goblin.artifacts import ResolvedArtifact
from classify_goblin.multimodal import prepare_video
from classify_goblin.multimodal.video import (
    FrameSamplerResult,
    VideoDecoderError,
    VideoDecoderTimeout,
    default_frame_sampler,
)

try:
    import subprocess
    _HAVE_FFMPEG = subprocess.run(
        ['ffprobe', '-version'], capture_output=True, timeout=5
    ).returncode == 0
except Exception:
    _HAVE_FFMPEG = False


def _artifact(data=b'\x00\x00\x00\x18ftypmp42'):
    return ResolvedArtifact('v1', 'video', 'video/mp4',
                           hashlib.sha256(data).hexdigest(), (), {}, data)


def _ok_sampler(frames):
    def frame_sampler(video_bytes, *, max_frames=16, timeout=10.0):
        return FrameSamplerResult({'duration': 1.0, 'frame_count': len(frames),
                                   'container': 'mp4', 'width': 4, 'height': 2},
                                  tuple(frames))
    return frame_sampler


class VideoPolicyTests(unittest.TestCase):
    def test_fail_closed_without_probe(self):
        payload = prepare_video(_artifact())
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'video_not_verified')
        self.assertEqual(payload.frames, ())
        self.assertNotIn('data', repr(payload))

    def test_ok_with_probe_and_no_sampler(self):
        payload = prepare_video(_artifact(), probe_verified=True)
        self.assertEqual(payload.status, 'ok')
        self.assertEqual(payload.capabilities.get('frame_sampling'), 'not_requested')
        self.assertEqual(payload.frames, ())

    def test_ok_with_probe_and_sampler(self):
        payload = prepare_video(_artifact(), probe_verified=True,
                               frame_sampler=_ok_sampler([(0, 0.0, 4, 2, b'f0'),
                                                        (1, 0.5, 4, 2, b'f1')]))
        self.assertEqual(payload.status, 'ok')
        self.assertEqual(len(payload.frames), 2)
        self.assertEqual(payload.features['sampled_frames'], 2)
        self.assertNotIn('data', repr(payload))

    def test_sampler_failure_is_partial(self):
        def frame_sampler(video_bytes, *, max_frames=16, timeout=10.0):
            raise RuntimeError('decode boom')
        payload = prepare_video(_artifact(), probe_verified=True,
                               frame_sampler=frame_sampler)
        self.assertEqual(payload.status, 'partial')
        self.assertEqual(payload.capabilities.get('frame_sampling'), 'extraction_failed')

    def test_oversize_bytes(self):
        from classify_goblin.multimodal import Limits
        payload = prepare_video(_artifact(b'x' * (9 * 1024 * 1024)), probe_verified=True,
                               limits=Limits(max_bytes=8 * 1024 * 1024))
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'invalid_or_oversize_bytes')

    def test_malformed_container_is_unavailable(self):
        # A payload that is not a recognised video container.
        payload = prepare_video(_artifact(b'NOTAVIDEOCONTAINER' * 10), probe_verified=True)
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'unsupported_container')

    def test_oversized_duration_is_unavailable(self):
        # A frame sampler that reports a duration over the 120 s limit.
        def frame_sampler(video_bytes, *, max_frames=16, timeout=10.0):
            raise VideoDecoderError('video_too_long')
        payload = prepare_video(_artifact(), probe_verified=True,
                               frame_sampler=frame_sampler)
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'video_too_long')

    def test_frame_budget_overflow_is_unavailable(self):
        def frame_sampler(video_bytes, *, max_frames=16, timeout=10.0):
            raise VideoDecoderError('too_many_frames')
        payload = prepare_video(_artifact(), probe_verified=True,
                               frame_sampler=frame_sampler)
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'too_many_frames')

    def test_frame_too_large_is_unavailable(self):
        # A frame exceeding the 4,096 x 2,160 resolution budget.
        def frame_sampler(video_bytes, *, max_frames=16, timeout=10.0):
            raise VideoDecoderError('frame_too_large')
        payload = prepare_video(_artifact(), probe_verified=True,
                               frame_sampler=frame_sampler)
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'frame_too_large')

    def test_timestamp_ordering_is_enforced(self):
        # A sampler that returns out-of-order timestamps is a validation failure.
        def frame_sampler(video_bytes, *, max_frames=16, timeout=10.0):
            return FrameSamplerResult({'duration': 1.0, 'frame_count': 2,
                                       'container': 'mp4', 'width': 4, 'height': 2},
                                      tuple([(1, 0.5, 4, 2, b'f1'), (0, 0.0, 4, 2, b'f0')]))
        payload = prepare_video(_artifact(), probe_verified=True,
                               frame_sampler=frame_sampler)
        self.assertEqual(payload.status, 'partial')
        self.assertEqual(payload.reason, 'frame_validation_failed')

    def test_parser_timeout_is_partial(self):
        def frame_sampler(video_bytes, *, max_frames=16, timeout=10.0):
            raise VideoDecoderTimeout('timeout')
        payload = prepare_video(_artifact(), probe_verified=True,
                               frame_sampler=frame_sampler)
        self.assertEqual(payload.status, 'partial')
        self.assertEqual(payload.capabilities.get('frame_sampling'), 'extraction_failed')

    def test_raw_media_never_in_repr(self):
        big = b'\x00\x00\x00\x18ftypmp42' + b'Z' * 1000
        payload = prepare_video(_artifact(big), probe_verified=True)
        self.assertNotIn('Z' * 100, repr(payload))
        self.assertEqual(payload.data, b'')


@unittest.skipUnless(_HAVE_FFMPEG, 'ffmpeg/ffprobe not available')
class DefaultFrameSamplerTests(unittest.TestCase):
    """Real ffprobe/ffmpeg against a tiny synthetic mp4 (no private media)."""

    @classmethod
    def setUpClass(cls):
        import subprocess
        import os
        cls.tmpdir = '/tmp/classify-goblin-mm-13-test'
        os.makedirs(cls.tmpdir, exist_ok=True)
        cls.mp4 = os.path.join(cls.tmpdir, 'tiny.mp4')
        subprocess.run(
            ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=red:s=64x64:d=1',
             '-f', 'lavfi', '-i', 'anullsrc=r=8000:cl=mono',
             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
             '-shortest', cls.mp4],
            capture_output=True, timeout=30)
        # A 130 s video for the oversized-duration test.
        cls.long_mp4 = os.path.join(cls.tmpdir, 'long.mp4')
        subprocess.run(
            ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=red:s=64x64:d=130',
             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', cls.long_mp4],
            capture_output=True, timeout=60)
        # A 10 s video at 2 fps = 20 sampled frames > 16 budget.
        cls.many_mp4 = os.path.join(cls.tmpdir, 'many.mp4')
        subprocess.run(
            ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=red:s=64x64:d=10',
             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', cls.many_mp4],
            capture_output=True, timeout=30)

    def _data(self, path):
        with open(path, 'rb') as f:
            return f.read()

    def test_default_sampler_ok(self):
        data = self._data(self.mp4)
        artifact = ResolvedArtifact('v1', 'video', 'video/mp4',
                                   hashlib.sha256(data).hexdigest(), (), {}, data)
        payload = prepare_video(artifact, probe_verified=True,
                               frame_sampler=default_frame_sampler)
        self.assertEqual(payload.status, 'ok')
        self.assertGreater(payload.features['sampled_frames'], 0)
        self.assertEqual(payload.features['duration'], 1.0)
        self.assertEqual(payload.features['frame_count'], 2)
        self.assertNotIn('data', repr(payload))

    def test_default_sampler_oversized_duration(self):
        data = self._data(self.long_mp4)
        artifact = ResolvedArtifact('v1', 'video', 'video/mp4',
                                   hashlib.sha256(data).hexdigest(), (), {}, data)
        payload = prepare_video(artifact, probe_verified=True,
                               frame_sampler=default_frame_sampler)
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'video_too_long')

    def test_default_sampler_frame_budget_overflow(self):
        data = self._data(self.many_mp4)
        artifact = ResolvedArtifact('v1', 'video', 'video/mp4',
                                   hashlib.sha256(data).hexdigest(), (), {}, data)
        payload = prepare_video(artifact, probe_verified=True,
                               frame_sampler=default_frame_sampler)
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'too_many_frames')

    def test_default_sampler_timeout(self):
        data = self._data(self.mp4)
        artifact = ResolvedArtifact('v1', 'video', 'video/mp4',
                                   hashlib.sha256(data).hexdigest(), (), {}, data)
        # A very short timeout forces a VideoDecoderTimeout.
        payload = prepare_video(artifact, probe_verified=True,
                               frame_sampler=lambda b, max_frames=16, timeout=0.001:
                               default_frame_sampler(b, max_frames=max_frames, timeout=timeout))
        self.assertEqual(payload.status, 'partial')
        self.assertEqual(payload.capabilities.get('frame_sampling'), 'extraction_failed')

    def test_default_sampler_unsupported_runtime(self):
        # Simulate ffprobe not being available.
        import classify_goblin.multimodal.video as video_mod
        orig = video_mod.subprocess.run
        def no_ffprobe(*a, **kw):
            raise FileNotFoundError('ffprobe')
        video_mod.subprocess.run = no_ffprobe
        try:
            data = self._data(self.mp4)
            artifact = ResolvedArtifact('v1', 'video', 'video/mp4',
                                       hashlib.sha256(data).hexdigest(), (), {}, data)
            payload = prepare_video(artifact, probe_verified=True,
                                   frame_sampler=default_frame_sampler)
        finally:
            video_mod.subprocess.run = orig
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'video_decode_failed')


if __name__ == '__main__':
    unittest.main()
