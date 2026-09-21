"""Bounded video adapter: probe-gated, frame-sampled, fail-closed.

Inputs are in-memory :class:`ResolvedArtifact` objects, never paths. The
payload is JSON-compatible: no raw video bytes, no paths, no filenames.
The optional ``frame_sampler`` callback is a trusted local adapter that
extracts a bounded number of frames and returns a
:class:`FrameSamplerResult`.
"""
from __future__ import annotations

import glob
import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

from ..artifacts import ResolvedArtifact
from .media import MediaPayload, _bad
from .preprocessing import DEFAULT_LIMITS, Limits

# Magic bytes for recognised video containers (checked before decode).
_VIDEO_MAGIC = (
    b'\x00\x00\x00\x18ftyp', b'\x00\x00\x00\x20ftyp', b'ftypisom',
    b'ftypmp42', b'ftypM4V ', b'ftypM4A ', b'\x1a\x45\xdf\xa3', b'RIFF',
)


class VideoDecoderError(Exception):
    """A video decode or validation failure with a stable reason code."""


class VideoDecoderTimeout(VideoDecoderError):
    """A video decode that exceeded its time budget."""


@dataclass(frozen=True)
class FrameSamplerResult:
    """Bounded metadata + sampled frames from a video decode."""
    metadata: dict
    frames: tuple


def _has_video_magic(data: bytes) -> bool:
    for magic in _VIDEO_MAGIC:
        if data[:16].find(magic) != -1:
            return True
    return False


def default_frame_sampler(video_bytes, *, max_frames=16, timeout=10.0):
    """Extract up to *max_frames* frames at 2 fps using ffprobe + ffmpeg.

    Raises :class:`VideoDecoderTimeout` on subprocess timeout and
    :class:`VideoDecoderError` on any other decode failure.
    """
    tmpdir = tempfile.mkdtemp(prefix='jev-video-')
    try:
        # 1. Probe for duration and metadata.
        try:
            probe = subprocess.run(
                ['ffprobe', '-v', 'error',
                 '-show_entries', 'format=duration',
                 '-show_entries', 'stream=codec_type,nb_frames,r_frame_rate',
                 '-of', 'default=noprint_wrappers=1',
                 '-'],
                input=video_bytes, capture_output=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise VideoDecoderTimeout('ffprobe timeout')
        except FileNotFoundError:
            raise VideoDecoderError('video_decode_failed')
        if probe.returncode != 0:
            raise VideoDecoderError('video_decode_failed')

        # Parse duration, width, height from ffprobe output.
        duration = 0.0
        width = 0
        height = 0
        for line in probe.stdout.decode('utf-8', errors='replace').splitlines():
            if line.startswith('duration='):
                try:
                    duration = float(line.split('=', 1)[1])
                except ValueError:
                    pass
            elif line.startswith('width='):
                try:
                    width = int(line.split('=', 1)[1])
                except ValueError:
                    pass
            elif line.startswith('height='):
                try:
                    height = int(line.split('=', 1)[1])
                except ValueError:
                    pass

        # 2. Enforce the duration budget before spending time on extraction.
        if duration > 120.0:
            raise VideoDecoderError('video_too_long')

        # 2b. Enforce the per-frame resolution budget (4,096 x 2,160).
        if width > 4096 or height > 2160:
            raise VideoDecoderError('frame_too_large')

        # 2c. Enforce the frame budget before extraction: at 2 fps the sampled
        # count is bounded by ceil(duration * 2). Failing here avoids decoding.
        if duration * 2.0 > max_frames:
            raise VideoDecoderError('too_many_frames')

        # 3. Extract frames at 2 fps.
        frame_pattern = os.path.join(tmpdir, 'frame_%03d.png')
        try:
            extract = subprocess.run(
                ['ffmpeg', '-v', 'error', '-i', 'pipe:0',
                 '-vf', 'fps=2', frame_pattern],
                input=video_bytes, capture_output=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise VideoDecoderTimeout('ffmpeg timeout')
        except FileNotFoundError:
            raise VideoDecoderError('video_decode_failed')
        if extract.returncode != 0:
            raise VideoDecoderError('video_decode_failed')

        # 3. Read back the extracted frames as (index, timestamp, w, h, bytes).
        frame_files = sorted(glob.glob(os.path.join(tmpdir, 'frame_*.png')))
        frames = []
        for index, path in enumerate(frame_files):
            with open(path, 'rb') as fh:
                frames.append((index, round(index * 0.5, 3), width, height, fh.read()))

        # 4. Validate the frame count against the budget.
        if len(frames) > max_frames:
            raise VideoDecoderError('too_many_frames')

        # 5. Build the metadata. frame_count is the number of sampled frames.
        metadata = {
            'duration': round(duration, 3),
            'frame_count': len(frames),
            'container': 'mp4' if b'ftyp' in video_bytes[:16] else 'unknown',
            'width': width,
            'height': height,
        }
        return FrameSamplerResult(metadata, tuple(frames))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def prepare_video(artifact, *, probe_verified=False, frame_sampler=None,
                 limits=DEFAULT_LIMITS):
    """Fail closed unless the local probe verified this video artifact."""
    data = artifact.data
    if not isinstance(data, bytes) or not (0 < len(data) <= limits.max_bytes):
        return _bad('video', 'video', artifact.mime, 'invalid_or_oversize_bytes')
    if not probe_verified:
        return _bad('video', 'video', artifact.mime, 'video_not_verified')
    if not _has_video_magic(data):
        return _bad('video', 'video', artifact.mime, 'unsupported_container')
    features = {'byte_size': len(data)}
    if frame_sampler is None:
        return MediaPayload('video', 'video', artifact.mime, 'ok', '',
                           features, {'frame_sampling': 'not_requested'}, data=b'')
    try:
        result = frame_sampler(data)
    except VideoDecoderTimeout:
        return MediaPayload('video', 'video', artifact.mime, 'partial',
                           'frame_extraction_failed', features,
                           {'frame_sampling': 'extraction_failed'}, data=b'')
    except VideoDecoderError as exc:
        reason = str(exc) if str(exc) in (
            'video_too_long', 'too_many_frames', 'frame_too_large', 'video_decode_failed'
        ) else 'frame_extraction_failed'
        if reason in ('video_too_long', 'too_many_frames', 'frame_too_large', 'video_decode_failed'):
            return _bad('video', 'video', artifact.mime, reason)
        return MediaPayload('video', 'video', artifact.mime, 'partial',
                           reason, features,
                           {'frame_sampling': 'extraction_failed'}, data=b'')
    except Exception:
        return MediaPayload('video', 'video', artifact.mime, 'partial',
                           'frame_extraction_failed', features,
                           {'frame_sampling': 'extraction_failed'}, data=b'')
    # Validate timestamp ordering.
    if len(result.frames) >= 2:
        # Frames are (index, timestamp, width, height, bytes) tuples.
        # If they are raw bytes, ordering is trivially satisfied.
        if all(isinstance(f, (tuple, list)) for f in result.frames):
            timestamps = [f[1] for f in result.frames]
            if any(timestamps[i] > timestamps[i + 1] for i in range(len(timestamps) - 1)):
                return MediaPayload('video', 'video', artifact.mime, 'partial',
                                   'frame_validation_failed', features,
                                   {'frame_sampling': 'extraction_failed'}, data=b'')
    merged_features = {**features, **result.metadata,
                      'sampled_frames': len(result.frames)}
    return MediaPayload('video', 'video', artifact.mime, 'ok', '',
                       merged_features, {'frame_sampling': 'available'},
                       frames=result.frames, data=b'')
