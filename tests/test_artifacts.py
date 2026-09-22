import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from classify_goblin import TypeSafeClient, AsyncTypeSafeClient, schema
from classify_goblin.artifacts import ArtifactBroker, ArtifactUnavailable, ArtifactSelectionError, validate
from classify_goblin.backends import RulesBackend
from test_contract import running, post, QUESTIONS


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'note.txt').write_bytes(b'hello')
        self.ref = dict(id='opaque-1', kind='text', mime='text/plain', path='note.txt',
                        sha256=hashlib.sha256(b'hello').hexdigest())
        self.body = dict(model='local-default', state='', questions=QUESTIONS, artifacts=[self.ref])

    def test_resolve_and_missing_roots(self):
        a = ArtifactBroker([str(self.root)]).resolve([self.ref])[0]
        self.assertEqual(a.data, b'hello')
        self.assertNotIn('hello', repr(a))
        self.assertEqual(a.usage(), dict(id='opaque-1', pages=[], truncated=False))
        with self.assertRaises(ArtifactUnavailable):
            ArtifactBroker([]).resolve([self.ref])

    def test_bad_references(self):
        for changes in ({'path': '/etc/passwd'}, {'path': '../note.txt'}, {'path': 'a/../note.txt'},
                        {'path': 'C:\\file'}, {'path': 'a\x00b'}, {'kind': []}, {'kind': 'audio'},
                        {'mime': 'application/pdf'}, {'sha256': 'bad'}, {'data': 'inline'},
                        {'id': 'path/a'}, {'pages': [1]}, {'crops': {'1': [[0, 0, 1, 1]]}}):
            with self.subTest(changes=changes), self.assertRaises(schema.ValidationError):
                validate([{**self.ref, **changes}])
        for items in (None, {}, [self.ref] * 9, [self.ref] * 2):
            with self.assertRaises(schema.ValidationError):
                validate(items)

    def test_pages_and_crops(self):
        pdf = {**self.ref, 'kind': 'pdf', 'mime': 'application/pdf', 'pages': [1, 3],
               'crops': {'3': [[0, .1, .5, 1]]}}
        validate([pdf])
        for changes in ({'pages': [True]}, {'pages': []}, {'pages': [0]}, {'pages': [1, 1]},
                        {'pages': [1, 2, 3, 4, 5]}, {'crops': {'2': [[0, 0, 1, 1]]}},
                        {'crops': {'3': [[0, 0, 1, 1]] * 5}},
                        {'crops': {'3': [[0, 0, float('nan'), 1]]}},
                        {'crops': {'3': [[1, 0, 0, 1]]}}):
            with self.subTest(changes=changes), self.assertRaises(schema.ValidationError):
                validate([{**pdf, **changes}])

    def test_filesystem_rejections(self):
        (self.root / 'link').symlink_to(self.root / 'note.txt')
        (self.root / 'dirlink').symlink_to(self.root, target_is_directory=True)
        os.mkfifo(self.root / 'fifo')
        for changes in ({'path': 'link'}, {'path': 'dirlink/note.txt'}, {'path': 'fifo'},
                        {'path': 'missing'}, {'sha256': '0' * 64}):
            with self.subTest(changes=changes), self.assertRaises(schema.ValidationError):
                ArtifactBroker([str(self.root)]).resolve([{**self.ref, **changes}])
        with patch('classify_goblin.artifacts.MAX_ARTIFACT_BYTES', 4):
            with self.assertRaises(schema.ValidationError):
                ArtifactBroker([str(self.root)]).resolve([self.ref])

    def test_http_and_sdk(self):
        class Backend(RulesBackend):
            def predict_artifacts(self, state, questions, artifacts):
                assert artifacts[0].data == b'hello'
                raw = self.predict(state, questions)
                raw['usage']['path'] = '/secret/path'
                raw['usage']['bytes'] = 'sensitive'
                return raw
        with patch.dict(os.environ, CLASSIFY_GOBLIN_ARTIFACT_ROOTS=str(self.root)), running(Backend()) as (server, url):
            client = TypeSafeClient(base_url=url, retries=0)
            result = client.systemOne(state='', questions=QUESTIONS, artifacts=[self.ref])
            self.assertEqual(set(result.usage), {'input_tokens', 'output_tokens', 'artifacts'})
            self.assertNotIn('path', schema.dumps(result))
            import asyncio
            result = asyncio.run(AsyncTypeSafeClient(base_url=url).systemOne(state='', questions=QUESTIONS, artifacts=[self.ref]))
            self.assertEqual(result.usage.artifacts[0].id, 'opaque-1')
            bad = copy.deepcopy(self.body)
            bad['artifacts'][0]['sha256'] = '0' * 64
            status, result = post(server, schema.dumps(bad))
            self.assertEqual(status, 422)
            self.assertIn('digest mismatch', result['error']['message'])
        with patch.dict(os.environ, CLASSIFY_GOBLIN_ARTIFACT_ROOTS=str(self.root)), running() as (server, url):
            self.assertEqual(post(server, schema.dumps(self.body))[0], 503)
            self.assertEqual(post(server, schema.dumps({**self.body, 'artifacts': []}))[0], 200)

    def test_decoded_selection_failure_is_422(self):
        class InvalidPage(RulesBackend):
            def predict_artifacts(self, *args):
                raise ArtifactSelectionError('/secret/document.pdf')
        with patch.dict(os.environ, CLASSIFY_GOBLIN_ARTIFACT_ROOTS=str(self.root)), running(InvalidPage()) as (server, url):
            status, body = post(server, schema.dumps(self.body))
            self.assertEqual(status, 422)
            self.assertNotIn('secret', schema.dumps(body))

    def test_backend_failure_is_safe(self):
        class Broken(RulesBackend):
            def predict_artifacts(self, *args):
                raise RuntimeError('/secret/path raw bytes')
        with patch.dict(os.environ, CLASSIFY_GOBLIN_ARTIFACT_ROOTS=str(self.root)), running(Broken()) as (server, url):
            status, body = post(server, schema.dumps(self.body))
            self.assertEqual(status, 503)
            self.assertNotIn('secret', schema.dumps(body))

    def test_no_root_stability_is_clean_503(self):
        # A6/T3: with no artifact roots configured the server stays stable —
        # a clean 503, no traceback, and no filesystem/secret leakage.
        with patch.dict(os.environ, {'CLASSIFY_GOBLIN_ARTIFACT_ROOTS': ''}):
            with running() as (server, url):
                status, body = post(server, schema.dumps(self.body))
                self.assertEqual(status, 503)
                self.assertIn('backend unavailable', body['error']['message'])
                self.assertNotIn('secret', schema.dumps(body))
                self.assertNotIn('CLASSIFY_GOBLIN_ARTIFACT_ROOTS', schema.dumps(body))


if __name__ == '__main__':
    unittest.main()
