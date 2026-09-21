"""Bounded media adapter: image/PDF first, video probe-gated, text/code passthrough.

Inputs are in-memory :class:`ResolvedArtifact` objects, never paths. The payload
is JSON-compatible: no raw video bytes, no paths, no filenames. Optional
callbacks (render_page, ocr_page, frame_sampler) are trusted local adapters.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from importlib import import_module
from io import BytesIO
import math
import warnings

from ..artifacts import ResolvedArtifact
from .preprocessing import DEFAULT_LIMITS, Limits, _text, preprocess_pdf

@dataclass(frozen=True)
class MediaPayload:
    kind: str
    modality: str
    mime: str
    status: str
    reason: str = ''
    features: dict = field(default_factory=dict)
    capabilities: dict = field(default_factory=dict)
    pages: tuple = ()
    frames: tuple = ()
    data: bytes = field(default=b'', repr=False)

    def usage(self):
        return {'kind': self.kind, 'modality': self.modality, 'status': self.status,
                'reason': self.reason, 'pages': list(self.pages)}


def _bad(kind, modality, mime, reason, **extra):
    return MediaPayload(kind, modality, mime, 'unavailable', reason, **extra)


def _jpeg_bytes(visual):
    buffer = BytesIO()
    if getattr(visual, 'mode', 'RGB') not in ('RGB', 'L'):
        visual = visual.convert('RGB')
    visual.save(buffer, format='JPEG', quality=90)
    return buffer.getvalue()


def _scale_for(width, height, budget):
    if budget <= 0 or width * height <= budget:
        return width, height
    factor = math.sqrt(budget / (width * height))
    return max(1, int(width * factor)), max(1, int(height * factor))


def _validate_crops(crops, width, height):
    """Return (status, reason, boxes) for the first page's unit-coordinate boxes."""
    for page_key, boxes in crops.items():
        for box in boxes:
            if (not isinstance(box, (list, tuple)) or len(box) != 4
                    or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in box)):
                return 'unavailable', 'crop_out_of_bounds', None
            x0, y0, x1, y1 = box
            if not (0 <= x0 <= x1 <= 1 and 0 <= y0 <= y1 <= 1):
                return 'unavailable', 'crop_out_of_bounds', None
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                return 'unavailable', 'crop_zero_area', None
    # Return the first box (a 4-tuple) for the adapter to apply.
    first = next(iter(crops.values()))[0]
    return 'ok', '', first


def prepare_media(artifact, *, modality=None, render_page=None, ocr_page=None,
                 frame_sampler=None, probe_verified=False, limits=DEFAULT_LIMITS):
    """Adapt a resolved artifact into a bounded, JSON-compatible payload."""
    kind = artifact.kind
    mod = modality or kind
    if mod == 'video':
        from .video import prepare_video as _prepare_video
        return _prepare_video(artifact, probe_verified=probe_verified,
                             frame_sampler=frame_sampler, limits=limits)
    if kind in ('text', 'code'):
        if not isinstance(artifact.data, bytes) or not artifact.data:
            return _bad(kind, mod, artifact.mime, 'empty_payload')
        return MediaPayload(kind, mod, artifact.mime, 'ok',
                           features={'characters': len(artifact.data)},
                           data=artifact.data)
    if kind == 'image':
        return _prepare_image(artifact, limits)
    if kind == 'pdf':
        return _prepare_pdf(artifact, render_page, ocr_page, limits)
    return _bad(kind, mod, artifact.mime, 'unsupported_kind')


def _prepare_image(artifact, limits):
    data = artifact.data
    if not isinstance(data, bytes) or not (0 < len(data) <= limits.max_bytes):
        return _bad('image', 'image', artifact.mime, 'invalid_or_oversize_bytes')
    # GIF is explicitly unsupported; gate on magic bytes before PIL decode so a
    # truncated GIF reports unsupported_format rather than a decode failure.
    if data[:6] in (b'GIF89a', b'GIF87a'):
        return _bad('image', 'image', artifact.mime, 'unsupported_format')
    try:
        image_module = import_module('PIL.Image')
    except ImportError:
        return _bad('image', 'image', artifact.mime, 'pillow_unavailable')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', image_module.DecompressionBombWarning)
            with image_module.open(BytesIO(data)) as img:
                if img.format not in {'PNG', 'JPEG', 'WEBP', 'BMP', 'TIFF'}:
                    return _bad('image', 'image', artifact.mime, 'unsupported_format')
                width, height = img.size
                if width <= 0 or height <= 0 or width * height > limits.max_pixels:
                    return _bad('image', 'image', artifact.mime, 'pixel_limit_exceeded')
                boxes = None
                if artifact.crops:
                    status, reason, boxes = _validate_crops(artifact.crops, width, height)
                    if status != 'ok':
                        return _bad('image', 'image', artifact.mime, reason)
                if boxes:
                    x0, y0, x1, y1 = boxes
                    cx0, cy0 = int(x0 * width), int(y0 * height)
                    cx1, cy1 = int(x1 * width), int(y1 * height)
                    if cx1 <= cx0 or cy1 <= cy0:
                        return _bad('image', 'image', artifact.mime, 'crop_zero_area')
                    img = img.crop((cx0, cy0, cx1, cy1))
                    width, height = img.size
                scaled_w, scaled_h = _scale_for(width, height, limits.max_model_pixels)
                if scaled_w != width or scaled_h != height:
                    img = img.resize((scaled_w, scaled_h))
                payload = _jpeg_bytes(img)
        return MediaPayload('image', 'image', 'image/jpeg', 'ok',
                           features={'width': width, 'height': height,
                                    'scaled_width': scaled_w, 'scaled_height': scaled_h,
                                    'byte_size': len(data),
                                    'aspect_ratio': round(width / height, 6)},
                           capabilities={'structure': 'available'},
                           data=payload)
    except image_module.DecompressionBombWarning:
        # A decompression bomb is a pixel-budget failure, not a decode error.
        return _bad('image', 'image', artifact.mime, 'pixel_limit_exceeded')
    except Exception:
        return _bad('image', 'image', artifact.mime, 'invalid_image')


def _prepare_pdf(artifact, render_page, ocr_page, limits):
    data = artifact.data
    if not isinstance(data, bytes) or not (0 < len(data) <= limits.max_bytes):
        return _bad('pdf', 'pdf', artifact.mime, 'invalid_or_oversize_bytes')
    if not data.startswith(b'%PDF-'):
        return _bad('pdf', 'pdf', artifact.mime, 'invalid_pdf_header')
    record = preprocess_pdf(data, selected_pages=list(artifact.pages) or None,
                           render_page=render_page, ocr_page=ocr_page, limits=limits)
    status = record['status']
    reason = record.get('reason', '')
    capabilities = {'render': 'hook_not_provided', 'ocr': 'hook_not_provided'}
    jpeg = b''
    render_ok = False
    for page in record.get('pages', []):
        capabilities['render'] = page['capabilities'].get('render', 'hook_not_provided')
        capabilities['ocr'] = page['capabilities'].get('ocr', 'hook_not_provided')
        if render_page is not None:
            try:
                visual = render_page(data, page['number'])
                if visual is not None:
                    jpeg = _jpeg_bytes(visual)
                    render_ok = True
            except Exception:
                capabilities['render'] = 'hook_failed'
    if render_ok:
        status = 'ok'
        reason = ''
    elif status == 'ok' and not jpeg and render_page is None:
        status = 'partial'
        reason = reason or 'no_visual_content'
    mime = 'image/jpeg' if jpeg else artifact.mime
    return MediaPayload('pdf', 'pdf', mime, status, reason,
                       features={'page_count': record['features'].get('page_count', 0),
                                'byte_size': len(data)},
                       capabilities=capabilities,
                       pages=tuple(artifact.pages), data=jpeg)
