"""Bounded, JSON-compatible artifact records with no binary payloads or filenames.

Inputs are in-memory bytes/text, never paths. Optional callbacks are trusted local
adapters: render_page(pdf_bytes, one_based_page) -> visual object;
ocr_page(pdf_bytes, one_based_page, visual_or_none) -> text;
visual_input(decoded_image) -> None. Callback outputs other than OCR text are
never recorded. Callbacks must enforce their own execution/memory limits. Library
parsers run in-process; byte/pixel/page limits are not a hostile-file sandbox.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
import re
import unicodedata
import warnings


@dataclass(frozen=True)
class Limits:
    max_bytes: int = 8 * 1024 * 1024
    max_text_chars: int = 8192
    max_pages: int = 100
    max_selected_pages: int = 8
    max_pixels: int = 16_000_000
    max_model_pixels: int = 0
    max_nodes: int = 20_000
    max_snippets: int = 16
    snippet_chars: int = 240

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value < (0 if name == 'max_model_pixels' else 1):
                raise ValueError('limits must be positive integers (max_model_pixels may be 0)')


DEFAULT_LIMITS = Limits()
# Remove POSIX, Windows and UNC absolute paths even when supplied inside content.
_PATH = re.compile(r"(?<![\w])(?:[A-Za-z]:[\\/]|\\\\|/)[^\s\"'<>`]*")


def _text(value, cap):
    value = unicodedata.normalize('NFKC', value)
    value = _PATH.sub('[path]', value)
    return ' '.join(''.join(c if c.isprintable() else ' ' for c in value).split())[:cap]


def _record(kind):
    return {'kind': kind, 'status': 'ok', 'features': {}, 'capabilities': {}}


def _cap(record, name, reason=None):
    record['capabilities'][name] = {'available': reason is None, 'reason': reason}
    if reason and record['status'] == 'ok':
        record['status'] = 'partial'


def _bad(record, reason):
    record['status'] = 'unavailable'
    record['reason'] = reason
    return record


def _valid_bytes(data, limits):
    return isinstance(data, bytes) and 0 < len(data) <= limits.max_bytes


def preprocess_text(text, *, limits=DEFAULT_LIMITS):
    record = _record('text')
    if not isinstance(text, str):
        return _bad(record, 'invalid_text')
    if len(text) > limits.max_bytes or len(text.encode('utf-8', errors='replace')) > limits.max_bytes:
        return _bad(record, 'input_too_large')
    normalized = _text(text, limits.max_bytes)
    record['text'] = normalized[:limits.max_text_chars]
    record['features'] = {'characters': len(text), 'truncated': len(normalized) > limits.max_text_chars}
    _cap(record, 'text')
    return record


def preprocess_code(source, *, extension='', diff=None, test_passed=None,
                    lint_errors=None, limits=DEFAULT_LIMITS):
    record = preprocess_text(source, limits=limits)
    record['kind'] = 'code'
    if record['status'] == 'unavailable':
        return record
    # Extension is a token, not a filename; never echo unknown caller input.
    ext = extension.lower().lstrip('.') if isinstance(extension, str) else ''
    languages = {'py': 'python', 'pyi': 'python', 'js': 'javascript', 'ts': 'typescript',
                 'tsx': 'typescript', 'jsx': 'javascript', 'rs': 'rust', 'go': 'go',
                 'java': 'java', 'c': 'c', 'cpp': 'cpp', 'sh': 'shell'}
    features = record['features']
    features.update(language=languages.get(ext, 'unknown'), lines=len(source.splitlines()))
    if test_passed is not None:
        if type(test_passed) is bool:
            features['test_passed'] = test_passed
        else:
            _cap(record, 'test_signal', 'invalid_test_signal')
    if lint_errors is not None:
        if type(lint_errors) is int and 0 <= lint_errors <= 1_000_000:
            features['lint_errors'] = lint_errors
        else:
            _cap(record, 'lint_signal', 'invalid_lint_signal')
    if diff is not None:
        if isinstance(diff, str) and len(diff) <= limits.max_bytes:
            lines = diff.splitlines()
            features['diff_added'] = sum(line.startswith('+') and not line.startswith('+++') for line in lines)
            features['diff_removed'] = sum(line.startswith('-') and not line.startswith('---') for line in lines)
        else:
            _cap(record, 'diff', 'invalid_or_oversize_diff')
    record['snippets'] = []
    if features['language'] != 'python':
        _cap(record, 'ast', 'unsupported_language')
        for number, line in enumerate(source.splitlines(), 1):
            if line.strip():
                record['snippets'].append({'line': number, 'text': _text(line, limits.snippet_chars)})
            if len(record['snippets']) >= limits.max_snippets:
                break
        return record
    try:
        tree = ast.parse(source, filename='<artifact>')
        nodes = []
        for node in ast.walk(tree):
            if len(nodes) >= limits.max_nodes:
                _cap(record, 'ast', 'node_limit_exceeded')
                return record
            nodes.append(node)
        features.update(ast_nodes=len(nodes), functions=sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in nodes),
                        classes=sum(isinstance(n, ast.ClassDef) for n in nodes),
                        imports=sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in nodes))
        lines = source.splitlines()
        for node in sorted((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))), key=lambda n: (n.lineno, n.col_offset))[:limits.max_snippets]:
            record['snippets'].append({'line': node.lineno, 'end_line': node.end_lineno,
                                       'column': node.col_offset, 'text': _text(lines[node.lineno - 1], limits.snippet_chars)})
        _cap(record, 'ast')
    except SyntaxError as error:
        features['syntax_error_line'] = error.lineno
        _cap(record, 'ast', 'syntax_error')
    except Exception:
        _cap(record, 'ast', 'parse_failed')
    return record


def preprocess_image(data, *, visual_input=None, limits=DEFAULT_LIMITS):
    record = _record('image')
    if not _valid_bytes(data, limits):
        return _bad(record, 'invalid_or_oversize_bytes')
    try:
        image_module = import_module('PIL.Image')
    except ImportError:
        return _bad(record, 'pillow_unavailable')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', image_module.DecompressionBombWarning)
            with image_module.open(BytesIO(data)) as img:
                width, height = img.size
                if img.format not in {'PNG', 'JPEG', 'WEBP', 'GIF', 'BMP', 'TIFF'}:
                    return _bad(record, 'unsupported_format')
                if width <= 0 or height <= 0 or width * height > limits.max_pixels:
                    return _bad(record, 'pixel_limit_exceeded')
                features = {'width': width, 'height': height, 'format': img.format,
                            'byte_size': len(data), 'aspect_ratio': round(width / height, 6),
                            'width_fraction': round(width / max(width, height), 6),
                            'height_fraction': round(height / max(width, height), 6),
                            'pixel_fraction': round(width * height / limits.max_pixels, 6),
                            'has_alpha': 'A' in img.getbands() or 'transparency' in img.info}
                img.verify()
        record['features'] = features
        _cap(record, 'structure')
        if visual_input is None:
            _cap(record, 'visual_input', 'hook_not_provided')
        else:
            try:
                with image_module.open(BytesIO(data)) as img:
                    img.load()
                    visual_input(img)
                _cap(record, 'visual_input')
            except Exception:
                _cap(record, 'visual_input', 'hook_or_decode_failed')
    except Exception:
        return _bad(record, 'invalid_image')
    return record


def preprocess_pdf(data, *, selected_pages=None, expected_page_count=None,
                   render_page=None, ocr_page=None, limits=DEFAULT_LIMITS):
    """Extract selected 1-based pages; default selects the first bounded page set.

    With no pypdf, explicit page selection permits render/OCR adapters to run,
    but page count and bounds remain explicitly unvalidated.
    """
    record = _record('pdf')
    record['pages'] = []
    if not _valid_bytes(data, limits):
        return _bad(record, 'invalid_or_oversize_bytes')
    if not data.startswith(b'%PDF-'):
        return _bad(record, 'invalid_pdf_header')
    if expected_page_count is not None and (type(expected_page_count) is not int or not 1 <= expected_page_count <= limits.max_pages):
        return _bad(record, 'invalid_expected_page_count')
    if selected_pages is not None:
        if (not isinstance(selected_pages, (list, tuple)) or not 1 <= len(selected_pages) <= limits.max_selected_pages
                or any(type(p) is not int or not 1 <= p <= limits.max_pages for p in selected_pages)):
            return _bad(record, 'invalid_selected_pages')
        selected_pages = sorted(set(selected_pages))
    reader = None
    try:
        pdf_module = import_module('pypdf')
    except ImportError:
        _cap(record, 'pdf_parser', 'pypdf_unavailable')
        _cap(record, 'page_count', 'pypdf_unavailable')
        _cap(record, 'metadata', 'pypdf_unavailable')
        if selected_pages is None:
            return _bad(record, 'explicit_pages_required_without_parser')
    else:
        try:
            reader = pdf_module.PdfReader(BytesIO(data))
            if reader.is_encrypted:
                return _bad(record, 'encrypted_pdf')
            count = len(reader.pages)
            if not 1 <= count <= limits.max_pages:
                return _bad(record, 'page_count_limit_exceeded')
            if expected_page_count is not None and count != expected_page_count:
                return _bad(record, 'page_count_mismatch')
            if selected_pages is not None and any(p > count for p in selected_pages):
                return _bad(record, 'page_out_of_range')
            selected_pages = selected_pages or list(range(1, min(count, limits.max_selected_pages) + 1))
            record['features'] = {'page_count': count, 'selected_count': len(selected_pages), 'selection_truncated': len(selected_pages) < count,
                                  'byte_size': len(data)}
            # Structural metadata only; author/title/producer strings may be private.
            metadata = reader.metadata or {}
            record['features']['has_metadata'] = bool(metadata)
            record['metadata'] = {
                'fields_present': [key[1:].lower() for key in
                                   ('/Title', '/Author', '/Subject', '/Keywords', '/Creator', '/Producer', '/CreationDate', '/ModDate')
                                   if key in metadata],
                'version': data[:8].decode('ascii') if re.fullmatch(rb'%PDF-[12]\.[0-9]', data[:8]) else 'unknown',
            }
            _cap(record, 'pdf_parser')
            _cap(record, 'metadata')
            _cap(record, 'page_count')
        except Exception:
            return _bad(record, 'pdf_parse_failed')
    remaining = limits.max_text_chars
    for number in selected_pages:
        page_record = {'number': number, 'text': '', 'capabilities': {}}
        record['pages'].append(page_record)
        text = ''
        if reader is not None:
            try:
                page = reader.pages[number - 1]
                text = page.extract_text() or ''
                page_record['capabilities']['text'] = 'available' if text.strip() else 'empty'
            except Exception:
                page_record['capabilities']['text'] = 'extraction_failed'
        else:
            page_record['capabilities']['text'] = 'parser_unavailable'
        visual = None
        if render_page is not None:
            try:
                visual = render_page(data, number)
                page_record['capabilities']['render'] = 'available' if visual is not None else 'empty'
            except Exception:
                page_record['capabilities']['render'] = 'hook_failed'
        else:
            page_record['capabilities']['render'] = 'hook_not_provided'
        if not text.strip():
            if ocr_page is None:
                page_record['capabilities']['ocr'] = 'hook_not_provided'
            else:
                try:
                    candidate = ocr_page(data, number, visual)
                    if not isinstance(candidate, str) or len(candidate) > limits.max_bytes:
                        raise ValueError('invalid OCR result')
                    text = candidate
                    page_record['capabilities']['ocr'] = 'available' if text.strip() else 'empty'
                except Exception:
                    page_record['capabilities']['ocr'] = 'hook_failed'
            if not text.strip():
                record['status'] = 'partial'
        normalized = _text(text[:limits.max_bytes], limits.max_bytes)
        page_record['text'] = normalized[:remaining]
        page_record['truncated'] = len(normalized) > remaining or len(text) > limits.max_bytes
        remaining -= len(page_record['text'])
    return record
