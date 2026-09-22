"""Qwen backend: manifest-gated, fail-closed, no raw bytes in the record."""
import io
import unittest

from classify_goblin.multimodal.qwen_backend import QwenBackend

try:
    import PIL.Image
except ImportError:
    PIL = None


def _png_bytes(size=(4, 2), color=(255, 0, 0)):
    buf = io.BytesIO()
    PIL.Image.new('RGB', size, color).save(buf, format='PNG')
    return buf.getvalue()


class QwenBackendTests(unittest.TestCase):
    def setUp(self):
        if PIL is None:
            self.skipTest('Pillow not installed')
        self.backend = QwenBackend(manifest_path='/nonexistent/manifest.json')

    def test_video_without_verified_manifest_is_unavailable(self):
        result = self.backend.run('video', b'\x00\x00\x00\x18ftypmp42', probe_verified=False)
        self.assertEqual(result.status, 'unavailable')
        self.assertEqual(result.reason, 'video_not_verified')
        self.assertNotIn('data', repr(result))

    def test_image_with_verified_manifest_is_ok(self):
        backend = QwenBackend(manifest_path='/nonexistent/manifest.json',
                             local_capabilities={'image': True})
        result = backend.run('image', _png_bytes(), probe_verified=True)
        self.assertEqual(result.status, 'ok')
        self.assertEqual(result.modality, 'image')
        self.assertNotIn('data', repr(result))

    def test_unknown_modality_is_unavailable(self):
        backend = QwenBackend(manifest_path='/nonexistent/manifest.json',
                             local_capabilities={'image': True})
        result = backend.run('hologram', b'x', probe_verified=True)
        self.assertEqual(result.status, 'unavailable')
        self.assertEqual(result.reason, 'modality_not_supported')

    def test_audio_is_unavailable(self):
        backend = QwenBackend(manifest_path='/nonexistent/manifest.json',
                             local_capabilities={'image': True})
        result = backend.run('audio', b'x', probe_verified=True)
        self.assertEqual(result.status, 'unavailable')
        self.assertEqual(result.reason, 'modality_not_supported')

    def test_empty_bytes_is_unavailable(self):
        backend = QwenBackend(manifest_path='/nonexistent/manifest.json',
                             local_capabilities={'image': True})
        result = backend.run('image', b'', probe_verified=True)
        self.assertEqual(result.status, 'unavailable')
        self.assertEqual(result.reason, 'empty_payload')

    def test_result_never_contains_raw_bytes(self):
        backend = QwenBackend(manifest_path='/nonexistent/manifest.json',
                             local_capabilities={'image': True})
        result = backend.run('image', _png_bytes(), probe_verified=True)
        self.assertNotIn(b'\x89PNG', repr(result).encode())


if __name__ == '__main__':
    unittest.main()
