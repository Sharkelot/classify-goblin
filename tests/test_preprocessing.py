import importlib.util
import json
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

from jev_laya_free.multimodal import (
    Limits, preprocess_code, preprocess_image, preprocess_pdf, preprocess_text,
)

MODULE = 'jev_laya_free.multimodal.preprocessing'


class TextCodeTests(unittest.TestCase):
    def test_normalization_bounds_and_privacy(self):
        record = preprocess_text('Ａ\n hello /home/private/data C:\\secret\\a.txt', limits=Limits(max_text_chars=80))
        self.assertEqual(record['text'], 'A hello [path] [path]')
        self.assertNotIn('/home', json.dumps(record))
        self.assertEqual(preprocess_text('abcdef', limits=Limits(max_text_chars=3))['text'], 'abc')
        self.assertTrue(preprocess_text('abcdef', limits=Limits(max_text_chars=3))['features']['truncated'])
        self.assertEqual(preprocess_text('éé', limits=Limits(max_bytes=3))['reason'], 'input_too_large')
        self.assertEqual(preprocess_text(None)['reason'], 'invalid_text')

    def test_python_ast_and_locations(self):
        source = 'import os\nclass C:\n    async def f(self):\n        return 1\n'
        record = preprocess_code(source, extension='.py', test_passed=True, lint_errors=2,
                                 diff='--- old\n+++ new\n-hello\n+world')
        self.assertEqual(record, preprocess_code(source, extension='.py', test_passed=True, lint_errors=2,
                                                diff='--- old\n+++ new\n-hello\n+world'))
        self.assertEqual(record['features']['functions'], 1)
        self.assertEqual(record['features']['classes'], 1)
        self.assertEqual(record['features']['imports'], 1)
        self.assertEqual(record['features']['diff_added'], 1)
        self.assertEqual(record['features']['diff_removed'], 1)
        self.assertEqual(record['snippets'][1]['line'], 3)
        self.assertEqual(record['snippets'][1]['column'], 4)
        self.assertEqual(record['capabilities']['ast']['reason'], None)

    def test_python_failures_and_generic_signals(self):
        self.assertEqual(preprocess_code('def (', extension='py')['capabilities']['ast']['reason'], 'syntax_error')
        record = preprocess_code('x = 1', extension='py', limits=Limits(max_nodes=1))
        self.assertEqual(record['capabilities']['ast']['reason'], 'node_limit_exceeded')
        record = preprocess_code('\nconst a = 1;\nconst b=2;', extension='js', test_passed='private', lint_errors=-1,
                                 limits=Limits(max_snippets=1, snippet_chars=5))
        self.assertEqual(record['features']['language'], 'javascript')
        self.assertEqual(record['snippets'], [{'line': 2, 'text': 'const'}])
        self.assertEqual(record['capabilities']['ast']['reason'], 'unsupported_language')
        self.assertNotIn('private', json.dumps(record))

    def test_limits_validation(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                Limits(max_bytes=value)


class PDFTests(unittest.TestCase):
    data = b'%PDF-1.4\nplaceholder'

    def reader(self, texts=('hello', 'world')):
        return SimpleNamespace(is_encrypted=False, metadata={'/Author': '/private/name'},
                               pages=[SimpleNamespace(extract_text=lambda text=text: text) for text in texts])

    def test_selection_count_and_global_text_bound(self):
        with patch(MODULE + '.import_module', return_value=SimpleNamespace(PdfReader=lambda _: self.reader())):
            record = preprocess_pdf(self.data, selected_pages=[2, 1, 2], expected_page_count=2,
                                    limits=Limits(max_text_chars=7))
            self.assertEqual([p['number'] for p in record['pages']], [1, 2])
            self.assertEqual([p['text'] for p in record['pages']], ['hello', 'wo'])
            self.assertTrue(record['pages'][1]['truncated'])
            self.assertNotIn('/private', json.dumps(record))
            self.assertEqual(preprocess_pdf(self.data, selected_pages=[3])['reason'], 'page_out_of_range')
            self.assertEqual(preprocess_pdf(self.data, expected_page_count=1)['reason'], 'page_count_mismatch')
            self.assertEqual(preprocess_pdf(self.data, limits=Limits(max_pages=1))['reason'], 'page_count_limit_exceeded')

    def test_missing_parser_and_ocr_hooks(self):
        with patch(MODULE + '.import_module', side_effect=ImportError('private path')):
            self.assertEqual(preprocess_pdf(self.data)['reason'], 'explicit_pages_required_without_parser')
            record = preprocess_pdf(self.data, selected_pages=[1], render_page=lambda data, page: b'private pixels',
                                    ocr_page=lambda data, page, visual: 'scanned /secret/file')
            self.assertEqual(record['pages'][0]['text'], 'scanned [path]')
            self.assertEqual(record['capabilities']['page_count']['reason'], 'pypdf_unavailable')
            self.assertNotIn('private', json.dumps(record))

    def test_failure_reasons_and_input_validation(self):
        for pages in ([], [0], [True], '1', [1] * 9):
            self.assertEqual(preprocess_pdf(self.data, selected_pages=pages)['reason'], 'invalid_selected_pages')
        self.assertEqual(preprocess_pdf(b'hello')['reason'], 'invalid_pdf_header')
        self.assertEqual(preprocess_pdf(self.data, expected_page_count=True)['reason'], 'invalid_expected_page_count')
        def fail(*args):
            raise RuntimeError('/private/file')
        with patch(MODULE + '.import_module', return_value=SimpleNamespace(PdfReader=lambda _: self.reader(('',)))):
            record = preprocess_pdf(self.data, render_page=fail, ocr_page=fail)
            self.assertEqual(record['pages'][0]['capabilities']['ocr'], 'hook_failed')
            self.assertNotIn('/private', json.dumps(record))
        with patch(MODULE + '.import_module', return_value=SimpleNamespace(PdfReader=fail)):
            self.assertEqual(preprocess_pdf(self.data)['reason'], 'pdf_parse_failed')

    @unittest.skipUnless(importlib.util.find_spec('pypdf'), 'optional pypdf not installed')
    def test_real_pdf(self):
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=200)
        stream = BytesIO()
        writer.write(stream)
        record = preprocess_pdf(stream.getvalue(), ocr_page=lambda *_: 'blank page')
        self.assertEqual(record['features']['page_count'], 1)
        self.assertEqual(record['pages'][0]['text'], 'blank page')


class ImageTests(unittest.TestCase):
    def test_missing_library_and_input_bounds(self):
        with patch(MODULE + '.import_module', side_effect=ImportError):
            self.assertEqual(preprocess_image(b'hello')['reason'], 'pillow_unavailable')
        for data in (b'', None, '/private/image.png', b'12345'):
            self.assertEqual(preprocess_image(data, limits=Limits(max_bytes=4))['reason'], 'invalid_or_oversize_bytes')

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'optional Pillow not installed')
    def test_real_image_structure_and_hooks(self):
        from PIL import Image
        stream = BytesIO()
        Image.new('RGBA', (4, 2)).save(stream, format='PNG')
        data = stream.getvalue()
        seen = []
        record = preprocess_image(data, visual_input=lambda img: seen.append(img.size))
        self.assertEqual(seen, [(4, 2)])
        self.assertEqual(record['features']['aspect_ratio'], 2)
        self.assertTrue(record['features']['has_alpha'])
        self.assertEqual(record['status'], 'ok')
        json.dumps(record)
        self.assertEqual(preprocess_image(data, limits=Limits(max_pixels=1))['reason'], 'pixel_limit_exceeded')
        self.assertEqual(preprocess_image(b'garbage')['reason'], 'invalid_image')
        def fail(img):
            raise RuntimeError('/private/image')
        record = preprocess_image(data, visual_input=fail)
        self.assertEqual(record['capabilities']['visual_input']['reason'], 'hook_or_decode_failed')
        self.assertNotIn('/private', json.dumps(record))


if __name__ == '__main__':
    unittest.main()
