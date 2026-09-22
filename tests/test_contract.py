import copy
import json
import os
import threading
import unittest
from contextlib import contextmanager
from http.client import HTTPConnection
from unittest.mock import patch

from classify_goblin import Choice, Score, Noul, TypeSafeClient, ClientError, ValidationError
from classify_goblin import schema
from classify_goblin.backends import RulesBackend, LayaBackend
from classify_goblin.server import LocalServer
from classify_goblin.workflow import decide, guard

QUESTIONS = {
    'route': Choice(instructions='Route', options={'code': 'source code', 'pdf': 'document'}),
    'quality': Score(instructions='Quality', criteria=['missing', 'partial', 'complete']),
    'review': Noul(instructions='Review?', criteria={'true': 'missing', 'false': 'complete'}),
}


@contextmanager
def running(backend=None, token=None):
    server = LocalServer(('127.0.0.1', 0), backend or RulesBackend(), token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def post(server, payload, headers=None, path='/v1/systemone'):
    conn = HTTPConnection('127.0.0.1', server.server_port, timeout=2)
    conn.request('POST', path, payload, headers or {'Content-Type': 'application/json'})
    reply = conn.getresponse()
    status, body = reply.status, json.loads(reply.read())
    conn.close()
    return status, body


class SchemaTests(unittest.TestCase):
    def test_arbitrary_state_and_structured_descriptions(self):
        for state in ('hello', {}, [1, True, None, {'data': [3.5]}]):
            body = schema.request({'model': 'local-default', 'state': state, 'questions': QUESTIONS})
            self.assertEqual(body['state'], state)
        questions = {'x': Choice(instructions={'question': ['where']}, criteria={'a': None, 'b': {'details': [1]}}),
                     's': Score(instructions=['rate'], criteria=[{'a': 'low'}, ['high']]),
                     'n': Noul(instructions='yes?', criteria={'true': ['yes']})}
        req = schema.request({'model': 'local-default', 'state': [], 'questions': questions})
        raw = RulesBackend().predict(req['state'], req['questions'])
        schema.answers(raw['answers'], req['questions'])

    def test_choice_aliases(self):
        for key in ('criteria', 'options'):
            for value in (['a', 'b'], {'a': None, 'b': 'B'}):
                body = schema.request({'model': 'local-default', 'state': '', 'questions': {'q': {'type': 'choice', 'instructions': 'pick', key: value}}})
                self.assertEqual(list(body['questions']['q']['criteria']), ['a', 'b'])

    def test_invalid_input(self):
        bad = [None, [], {}, {'model': 'local-default', 'state': True, 'questions': QUESTIONS},
               {'model': 'local-default', 'state': {}, 'questions': {}}, {'model': 'local-default', 'state': {}, 'questions': QUESTIONS, 'extra': 1},
               {'model': 'local-default', 'state': float('nan'), 'questions': QUESTIONS},
               {'model': 'local-default', 'state': 'x'*65536, 'questions': QUESTIONS}]
        bad_q = [{'type': 'freeform', 'instructions': 'x'},
                 {'type': 'noul'}, {'type': 'noul', 'instructions': 'x', 'options': ['a']},
                 {'type': 'noul', 'instructions': 'x', 'criteria': {'maybe': 'x'}},
                 {'type': 'choice', 'instructions': 'x', 'options': ['a', 'a']},
                 {'type': 'choice', 'instructions': 'x', 'options': ['a'], 'criteria': {'a': None}},
                 {'type': 'score', 'instructions': 'x', 'criteria': ['one']},
                 {'type': 'score', 'instructions': 'x', 'criteria': ['x']*11},
                 {'type': 'choice', 'instructions': 'x', 'options': [True]}]
        bad += [{'model': 'local-default', 'state': {}, 'questions': {'q': q}} for q in bad_q]
        for payload in bad:
            with self.subTest(payload=str(payload)[:100]), self.assertRaises(ValidationError):
                schema.request(payload)

    def test_bad_json_and_depth(self):
        for raw in (b'{"x":1,"x":2}', b'{"n": NaN}', b'{"n": Infinity}', b'\xff', b'['*30 + b']'*30):
            with self.assertRaises(ValidationError):
                schema.loads(raw)

    def test_answer_validation(self):
        questions = schema.request({'model': 'local-default', 'state': '', 'questions': QUESTIONS})['questions']
        good = RulesBackend().predict('', questions)['answers']
        mutations = [lambda x: x.pop('review'),
                     lambda x: x['route'].update(choice='unknown'),
                     lambda x: x['route'].update(confidence=float('nan')),
                     lambda x: x['route']['probabilities'].update(code=1.1),
                     lambda x: x['review'].update(noul=True),
                     lambda x: x['quality'].update(score=0),
                     lambda x: x['quality'].update(legend={})]
        for mutate in mutations:
            value = copy.deepcopy(good)
            mutate(value)
            with self.assertRaises(ValidationError):
                schema.answers(value, questions)


class HTTPTests(unittest.TestCase):
    def test_roundtrip_and_builders(self):
        with running() as (_, url):
            with TypeSafeClient(base_url=url) as client:
                for state in ('code', {'data': [True, None, 2]}, ['code', 'missing']):
                    result = client.systemOne(state=state, questions=QUESTIONS)
                    self.assertEqual(set(result.answers), set(QUESTIONS))
                    self.assertEqual(result.model, 'local-rules-v1')
                    self.assertEqual(set(result.answers.review), {'type', 'noul'})
                    self.assertEqual(set(result.answers.quality.legend), {'0', '1', '2'})
                    self.assertEqual(result.usage.input_tokens, 0)
                    self.assertTrue(result.request_id)

    def test_choice_and_score_label_order_preserved_roundtrip(self):
        # S6/T3: criteria/label order is preserved server->client byte-for-byte.
        # Choice options are an ordered dict; score levels are an ordered array
        # with an ordinal-string legend. Neither may be reordered or sorted.
        ordered = {
            'pick': Choice(instructions='Pick',
                           options={'zeta': 'last', 'alpha': 'first', 'mid': 'middle'}),
            'grade': Score(instructions='Grade',
                          criteria=['zero', 'one', 'two']),
        }
        with running() as (_, url):
            with TypeSafeClient(base_url=url) as client:
                result = client.systemOne(state='code', questions=ordered)
                # Choice: probability keys keep the request insertion order.
                self.assertEqual(list(result.answers.pick.probabilities),
                                ['zeta', 'alpha', 'mid'])
                # Score: legend is keyed by ordinal string, in criteria order.
                self.assertEqual(list(result.answers.grade.legend), ['0', '1', '2'])
                self.assertEqual(result.answers.grade.legend['0'], 'zero')
                self.assertEqual(result.answers.grade.legend['2'], 'two')
                # Positional targets (score) are not reordered: weighted mean
                # must match the ordinal positions, not a re-sorted set.
                self.assertEqual(result.answers.grade.score,
                                sum(int(k) * v for k, v in
                                    result.answers.grade.probabilities.items()))

    def test_auth_and_errors(self):
        with running(token='test-key') as (server, url):
            body = schema.dumps({'model': 'local-default', 'state': '', 'questions': QUESTIONS})
            self.assertEqual(post(server, body)[0], 401)
            with self.assertRaises(ClientError) as cm:
                TypeSafeClient(base_url=url, api_key='wrong').system_one(state='', questions=QUESTIONS)
            self.assertEqual(cm.exception.status, 401)
            result = TypeSafeClient(base_url=url, api_key='test-key').system_one(state='', questions=QUESTIONS)
            self.assertTrue(result.answers)
        with running() as (server, url):
            for payload, status in [('{}', 422), ('{"state":0,"state":1}', 422), ('x'*65537, 413)]:
                self.assertEqual(post(server, payload)[0], status)
            self.assertEqual(post(server, '{}', path='/missing')[0], 404)
            self.assertEqual(post(server, '{}', {'Content-Type': 'text/plain'})[0], 415)
            self.assertEqual(post(server, '{}', {'Content-Type': 'application/json', 'Transfer-Encoding': 'chunked'})[0], 400)
            with self.assertRaises(ClientError) as cm:
                TypeSafeClient(base_url=url, model='classify-goblin-latest', retries=0).system_one(state='', questions=QUESTIONS)
            self.assertEqual(cm.exception.status, 422)

    def test_invalid_backend_is_not_success(self):
        class Bad(RulesBackend):
            def predict(self, *args):
                return {'answers': {}, 'usage': {'input_tokens': 0, 'output_tokens': 0}}
        with running(Bad()) as (_, url), self.assertRaises(ClientError) as cm:
            TypeSafeClient(base_url=url, retries=0).system_one(state='', questions=QUESTIONS)
        self.assertEqual(cm.exception.status, 502)

    def test_retry_transient_and_busy(self):
        class Flaky(RulesBackend):
            calls = 0
            def predict(self, *args):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError('private detail must not leak')
                return super().predict(*args)
        backend = Flaky()
        with running(backend) as (server, url):
            result = TypeSafeClient(base_url=url, retries=1).system_one(state='', questions=QUESTIONS)
            self.assertTrue(result.answers)
            self.assertEqual(backend.calls, 2)
            server.inference_lock.acquire()
            try:
                status, body = post(server, schema.dumps({'model': 'local-default', 'state': '', 'questions': QUESTIONS}))
                self.assertEqual(status, 529)
                self.assertNotIn('answers', body)
            finally:
                server.inference_lock.release()

    def test_framing_and_redirect_rejection(self):
        with running() as (server, _):
            for length in ('9'*100, '²'):
                status, _ = post(server, '{}', {'Content-Type': 'application/json', 'Content-Length': length})
                self.assertEqual(status, 400)
        from classify_goblin.client import NoRedirect
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, '', {}, 'http://example.com'))

    def test_client_local_only(self):
        for url in ('https://api.typesafe.ai', 'http://example.com', 'http://127.0.0.1/x', 'http://key@localhost'):
            with self.assertRaises(ValidationError):
                TypeSafeClient(base_url=url)
        for kwargs in ({'timeout': 0}, {'retries': -1}, {'api_key': 'bad\nheader'}):
            with self.assertRaises(ValidationError):
                TypeSafeClient(**kwargs)

    def test_client_rejects_bad_response_without_retry(self):
        from io import BytesIO
        from email.message import Message
        body = BytesIO(b'{"answers":{}}')
        body.headers = Message()
        body.headers['Content-Type'] = 'application/json'
        client = TypeSafeClient()
        with patch.object(client.opener, 'open', return_value=body) as op, self.assertRaises(ClientError):
            client.system_one(state='', questions=QUESTIONS)
        self.assertEqual(op.call_count, 1)

    def test_client_timeout_retries(self):
        client = TypeSafeClient(retries=1)
        with patch.object(client.opener, 'open', side_effect=TimeoutError) as op, self.assertRaises(ClientError):
            client.system_one(state='', questions=QUESTIONS)
        self.assertEqual(op.call_count, 2)


class BackendWorkflowTests(unittest.TestCase):
    def test_rules_repeatable_and_question_name_independent(self):
        backend = RulesBackend()
        q = schema.request({'model': 'local-default', 'state': '', 'questions': QUESTIONS})['questions']
        self.assertEqual(backend.predict('code', q), backend.predict('code', q))
        self.assertEqual(backend.predict('code', {'a': q['route']})['answers']['a'],
                         backend.predict('code', {'b': q['route']})['answers']['b'])

    def test_laya_local_path_and_translation_without_weights(self):
        import tempfile
        import types
        class Agent:
            def predict(self, state, questions):
                self.questions = questions
                raw = RulesBackend().predict(state, questions)
                raw['answers']['review']['confidence'] = 0.9
                raw['answers']['review']['action'] = {'execute': True}
                return raw
        agent = Agent()
        fake = types.SimpleNamespace(load=lambda path, device: agent)
        with tempfile.TemporaryDirectory() as path, patch.dict('sys.modules', {'laya': fake}), patch.dict(os.environ, {}, clear=False):
            backend = LayaBackend(path)
            self.assertEqual(os.environ['HF_HUB_OFFLINE'], '1')
            q = schema.request({'model': 'local-default', 'state': '', 'questions': QUESTIONS})['questions']
            raw = backend.predict('', q)
            self.assertEqual(set(raw['answers']['review']), {'type', 'noul'})
            q['route']['criteria'] = dict.fromkeys(map(str, range(9)))
            with self.assertRaises(ValidationError):
                backend.predict('', q)
        with self.assertRaises(ValidationError):
            LayaBackend('/definitely/not/a/local/model')

    def test_workflow_precedence_and_routes(self):
        for state, decision in [({'terminal': True, 'same_action_streak': 9}, 'terminal'),
                                ({'same_action_streak': 3}, 'stop'), ({'same_action_streak': 2}, 'review'),
                                ({'context_compactions': 2}, 'stop'), ({'same_tool_failures': 3}, 'stop'),
                                ({'same_action_streak': 9, 'artifact_delta': True}, 'allow')]:
            self.assertEqual(guard(state)['decision'], decision)
        for state, hand in [({'modality': 'code'}, 'inspect_code'), ({'modality': 'code', 'test_needed': True}, 'run_test'),
                            ({'modality': 'pdf'}, 'extract_pdf_text'), ({'modality': 'pdf', 'extraction_quality': 'partial'}, 'render_pdf_page'),
                            ({'modality': 'image'}, 'inspect_image'),
                            ({'modality': 'video'}, 'inspect_video'),
                            ({'modality': 'audio'}, 'transcribe_audio'),
                            ({'evidence_sufficient': True, 'source_grounded': True, 'source_digest': 'hash', 'location': 'p1'}, 'synthesize')]:
            self.assertEqual(decide(state)['route'], {'hand': hand, 'executor': 'qwen'})
        self.assertEqual(decide({'modality': 'image', 'same_action_streak': 3})['route']['executor'], None)

    def test_workflow_model_cannot_override(self):
        class Adversarial:
            calls = 0
            def system_one(self, **kwargs):
                self.calls += 1
                return {'answers': {'next_hand': {'choice': 'stop'}, 'needs_review': {'noul': 1}}}
        client = Adversarial()
        fresh = decide({'modality': 'code'}, client)
        self.assertEqual(fresh['gate']['decision'], 'allow')
        self.assertEqual(fresh['route']['hand'], 'inspect_code')
        stopped = decide({'same_action_streak': 3}, client)
        self.assertEqual(stopped['gate']['decision'], 'stop')
        self.assertEqual(client.calls, 1)
        with patch.object(client, 'system_one', side_effect=ClientError('offline')):
            self.assertEqual(decide({}, client)['advisory']['status'], 'unavailable')

    def test_workflow_full_fingerprints_and_strict_flags(self):
        state = {'same_action_streak': 3, 'result': 'x'*2000+'a', 'previous_result': 'x'*2000+'b'}
        self.assertEqual(guard(state)['decision'], 'allow')
        for state in ({'terminal': 'false'}, {'same_action_streak': True}, {'modality': 'hologram'}):
            with self.assertRaises(ValidationError):
                decide(state)


if __name__ == '__main__':
    unittest.main()
