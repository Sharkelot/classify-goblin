"""Bounded media adapter: image/PDF first, video probe-gated, text/code passthrough."""
import hashlib
import io
import unittest

from jev_laya_free.artifacts import ResolvedArtifact
from jev_laya_free.multimodal import Limits, prepare_media, prepare_video

try:
    import PIL.Image
except ImportError:
    PIL = None


def _image_bytes(size=(4, 2), color=(255, 0, 0)):
    if PIL is None:
        raise unittest.SkipTest('Pillow not installed')
    buf = io.BytesIO()
    PIL.Image.new('RGB', size, color).save(buf, format='PNG')
    return buf.getvalue()


def _artifact(kind, data, mime, **extra):
    digest = hashlib.sha256(data if data is not None else b'').hexdigest()
    return ResolvedArtifact('a1', kind, mime, digest,
                           tuple(extra.get('pages', [])), extra.get('crops', {}), data)


class ImageMediaTests(unittest.TestCase):
    def setUp(self):
        if PIL is None:
            self.skipTest('Pillow not installed')

    def test_small_image_scales_and_encodes_jpeg(self):
        data = _image_bytes((4, 2))
        payload = prepare_media(_artifact('image', data, 'image/png'))
        self.assertEqual(payload.status, 'ok')
        self.assertEqual(payload.kind, 'image')
        self.assertEqual(payload.mime, 'image/jpeg')
        self.assertIsInstance(payload.data, bytes)
        self.assertTrue(payload.data.startswith(b'\xff\xd8'), 'JPEG SOI expected')
        self.assertEqual(payload.features['width'], 4)
        self.assertEqual(payload.features['height'], 2)
        self.assertNotIn('data', repr(payload))

    def test_crop_applied(self):
        data = _image_bytes((10, 10))
        payload = prepare_media(_artifact('image', data, 'image/png', crops={1: [[0, 0, 0.5, 1]]}))
        self.assertEqual(payload.status, 'ok')
        self.assertEqual(payload.features['width'], 5)
        self.assertEqual(payload.features['height'], 10)

    def test_crop_out_of_bounds(self):
        data = _image_bytes((10, 10))
        payload = prepare_media(_artifact('image', data, 'image/png', crops={1: [[0.5, 0, 1.5, 1]]}))
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'crop_out_of_bounds')

    def test_crop_zero_area(self):
        data = _image_bytes((10, 10))
        payload = prepare_media(_artifact('image', data, 'image/png', crops={1: [[0, 0, 0, 1]]}))
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'crop_zero_area')

    def test_oversized_image_scaled_to_budget(self):
        limits = Limits(max_pixels=16_000_000, max_model_pixels=1000)
        data = _image_bytes((50, 20))
        payload = prepare_media(_artifact('image', data, 'image/png'), limits=limits)
        self.assertEqual(payload.status, 'ok')
        self.assertLessEqual(payload.features['scaled_width'] * payload.features['scaled_height'], 1000)
        self.assertEqual(payload.features['width'], 50)

    def test_pixel_limit_exceeded(self):
        limits = Limits(max_pixels=100)
        data = _image_bytes((20, 10))
        payload = prepare_media(_artifact('image', data, 'image/png'), limits=limits)
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'pixel_limit_exceeded')

    def test_unsupported_format(self):
        payload = prepare_media(_artifact('image', b'GIF89a\x01\x00', 'image/gif'))
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'unsupported_format')

    def test_invalid_bytes(self):
        for data in (None, b'', b'not-an-image'):
            payload = prepare_media(_artifact('image', data, 'image/png'))
            self.assertEqual(payload.status, 'unavailable')
            self.assertIn(payload.reason, ('invalid_or_oversize_bytes', 'invalid_image'))

    def test_oversize_bytes(self):
        limits = Limits(max_bytes=10)
        payload = prepare_media(_artifact('image', b'x' * 11, 'image/png'), limits=limits)
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'invalid_or_oversize_bytes')

    def test_no_paths_or_bytes_in_payload(self):
        data = _image_bytes((4, 2))
        payload = prepare_media(_artifact('image', data, 'image/png'))
        self.assertNotIn('ResolvedArtifact', repr(payload))
        for field in (payload.features, payload.capabilities):
            for value in field.values():
                if isinstance(value, (str, bytes)):
                    self.assertNotIn(b'\xff\xd8', value if isinstance(value, bytes) else value.encode())


class PDFMediaTests(unittest.TestCase):
    def test_pdf_without_render_or_ocr_is_partial(self):
        data = b'%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF'
        payload = prepare_media(_artifact('pdf', data, 'application/pdf', pages=[1]))
        self.assertIn(payload.status, ('partial', 'unavailable'))
        self.assertEqual(payload.kind, 'pdf')
        self.assertEqual(payload.pages, (1,))
        self.assertEqual(payload.data, b'')

    def test_pdf_with_render_hook_is_ok(self):
        data = b'%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF'
        if PIL is None:
            self.skipTest('Pillow not installed')

        def render_page(pdf_bytes, page):
            return PIL.Image.new('RGB', (4, 2), (0, 255, 0))

        payload = prepare_media(_artifact('pdf', data, 'application/pdf', pages=[1]), render_page=render_page)
        self.assertEqual(payload.status, 'ok')
        self.assertTrue(payload.data.startswith(b'\xff\xd8'))
        self.assertEqual(payload.mime, 'image/jpeg')

    def test_pdf_render_hook_failure_is_partial(self):
        data = b'%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF'

        def render_page(pdf_bytes, page):
            raise RuntimeError('boom')

        payload = prepare_media(_artifact('pdf', data, 'application/pdf', pages=[1]), render_page=render_page)
        self.assertEqual(payload.status, 'partial')
        self.assertEqual(payload.capabilities.get('render'), 'hook_failed')

    def test_pdf_invalid_header(self):
        payload = prepare_media(_artifact('pdf', b'not-a-pdf', 'application/pdf', pages=[1]))
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'invalid_pdf_header')

    def test_pdf_oversize_bytes(self):
        limits = Limits(max_bytes=10)
        payload = prepare_media(_artifact('pdf', b'%PDF-' + b'x' * 20, 'application/pdf', pages=[1]), limits=limits)
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'invalid_or_oversize_bytes')


class TextCodeMediaTests(unittest.TestCase):
    def test_text_passthrough(self):
        payload = prepare_media(_artifact('text', b'hello world', 'text/plain'))
        self.assertEqual(payload.status, 'ok')
        self.assertEqual(payload.data, b'hello world')
        self.assertEqual(payload.mime, 'text/plain')

    def test_code_passthrough(self):
        payload = prepare_media(_artifact('code', b'def f(): return 1', 'text/x-python'))
        self.assertEqual(payload.status, 'ok')
        self.assertEqual(payload.data, b'def f(): return 1')

    def test_video_unavailable_without_probe(self):
        payload = prepare_media(_artifact('image', b'x', 'image/png'), modality='video')
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'video_not_verified')


class VideoMediaTests(unittest.TestCase):
    def test_video_fail_closed_without_probe(self):
        data = b'\x00\x00\x00\x18ftypmp42'
        payload = prepare_video(_artifact('image', data, 'image/png'))
        self.assertEqual(payload.status, 'unavailable')
        self.assertEqual(payload.reason, 'video_not_verified')

    def test_video_ok_with_probe(self):
        data = b'\x00\x00\x00\x18ftypmp42'
        payload = prepare_video(_artifact('image', data, 'image/png'), probe_verified=True)
        self.assertEqual(payload.status, 'ok')
        self.assertEqual(payload.kind, 'video')
        self.assertEqual(payload.data, b'')


class DecompressionBombTests(unittest.TestCase):
    """P5: a small high-ratio PNG that decodes to >16M pixels must fail closed."""

    def test_bomb_png_rejected(self):
        if PIL is None:
            self.skipTest('Pillow not installed')
        from jev_laya_free.multimodal.preprocessing import preprocess_image
        # 4096x4096 = 16,777,216 pixels > 16,000,000 default budget.
        # A 1x1 image with a huge IHDR is not valid; use a real large PNG.
        # Generate a 4096x4096 PNG in-memory (single-color, compresses tiny).
        buf = io.BytesIO()
        PIL.Image.new('RGB', (4096, 4096), (0, 0, 0)).save(buf, format='PNG')
        data = buf.getvalue()
        self.assertLess(len(data), 1024 * 1024, 'fixture should be small on disk')
        record = preprocess_image(data)
        self.assertEqual(record['status'], 'unavailable')
        self.assertEqual(record.get('reason'), 'pixel_limit_exceeded')


if __name__ == '__main__':
    unittest.main()
