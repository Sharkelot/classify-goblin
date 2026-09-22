"""Exercise the official quickstart import/call/access shape using synthetic inputs."""
import asyncio
import os
import unittest
from unittest.mock import patch

from typesafe_sdk import (
    AsyncTypeSafeClient, Choice, Noul, Score, TypeSafeClient,
    ClientError, ValidationError,
)
from classify_goblin import schema
from test_contract import running, post


def questions():
    return {
        'billing': Noul(instructions='Does the record mention an invoice?'),
        'tone': Choice(instructions='Select a tone', criteria={'calm': None, 'angry': None}),
        'urgency': Score(instructions='Rate urgency', criteria=['routine', 'soon', 'urgent']),
    }


class ShimTests(unittest.TestCase):
    def check_response(self, response):
        self.assertEqual(set(response.answers), {'billing', 'tone', 'urgency'})
        self.assertEqual(set(response.choices), {'tone'})
        self.assertEqual(set(response.scores), {'urgency'})
        self.assertEqual(set(response.nouls), {'billing'})
        self.assertIs(response.choices['tone'], response.answers['tone'])
        self.assertEqual(response.choices['tone'].choice, response['answers']['tone']['choice'])
        self.assertIsInstance(response.scores['urgency'].score, (float, int))
        self.assertIsInstance(response.nouls['billing'].noul, (float, int))
        self.assertEqual(set(response.nouls['billing']), {'type', 'noul'})
        self.assertEqual(set(response), {'model', 'answers', 'usage', 'request_id'})
        self.assertEqual(response.model, 'local-rules-v1')
        self.assertEqual(response.usage.input_tokens, 0)
        self.assertTrue(response.request_id)

    def test_sync_quickstart_import_shape_and_env(self):
        with running(token='compat-key') as (_, url), patch.dict(os.environ, {
            'TYPESAFE_BASE_URL': url, 'TYPESAFE_API_KEY': 'compat-key',
        }, clear=True):
            with TypeSafeClient() as client:
                response = client.system_one(state={'document': 'urgent invoice'}, questions=questions())
            self.check_response(response)

    def test_async_quickstart_import_shape_and_alias(self):
        async def run():
            async with AsyncTypeSafeClient() as client:
                for call in (client.system_one, client.systemOne):
                    response = await call(state=['urgent invoice'], questions=questions())
                    self.check_response(response)
        with running(token='compat-key') as (_, url), patch.dict(os.environ, {
            'TYPESAFE_BASE_URL': url, 'TYPESAFE_API_KEY': 'compat-key',
        }, clear=True):
            asyncio.run(run())

    def test_env_precedence_and_local_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            client = TypeSafeClient()
            self.assertEqual(client.url, 'http://127.0.0.1:8093/v1/systemone')
            self.assertEqual(client.model, 'local-default')
        with patch.dict(os.environ, {'TYPESAFE_BASE_URL': 'https://api.typesafe.ai',
                                    'TYPESAFE_API_KEY': 'compat', 'CLASSIFY_GOBLIN_LOCAL_API_KEY': 'local',
                                    'TYPESAFE_DEFAULT_MODEL': 'classify-goblin-latest'}, clear=True):
            with self.assertRaises(ValidationError):
                TypeSafeClient()
            client = TypeSafeClient(base_url='http://localhost:8093')
            self.assertEqual(client.api_key, 'local')
            self.assertEqual(client.model, 'local-default')
            self.assertEqual(TypeSafeClient(base_url='http://localhost', api_key='explicit').api_key, 'explicit')

    def test_legacy_local_api_key_is_only_a_fallback(self):
        with patch.dict(os.environ, {'JEV_LOCAL_API_KEY': 'legacy'}, clear=True):
            self.assertEqual(TypeSafeClient().api_key, 'legacy')
        with patch.dict(os.environ, {
            'JEV_LOCAL_API_KEY': 'legacy',
            'CLASSIFY_GOBLIN_LOCAL_API_KEY': 'current',
        }, clear=True):
            self.assertEqual(TypeSafeClient().api_key, 'current')

    def test_required_wire_model_and_hosted_model_rejection(self):
        with running() as (server, url):
            status, body = post(server, schema.dumps({'state': '', 'questions': questions()}))
            self.assertEqual(status, 422)
            self.assertNotIn('answers', body)
            with self.assertRaises(ClientError) as cm:
                TypeSafeClient(base_url=url, retries=0).system_one(state='', questions=questions(), model='classify-goblin-latest')
            self.assertEqual(cm.exception.status, 422)
            async def run():
                async with AsyncTypeSafeClient(base_url=url, retries=0) as client:
                    with self.assertRaises(ClientError) as cm:
                        await client.system_one(state='', questions=questions(), model='classify-goblin-latest')
                    self.assertEqual(cm.exception.status, 422)
            asyncio.run(run())

    def test_async_transport_does_not_block_event_loop(self):
        import threading
        started, release = threading.Event(), threading.Event()
        client = AsyncTypeSafeClient(base_url='http://127.0.0.1:8093')
        def blocking(**kwargs):
            started.set()
            if not release.wait(2):
                raise RuntimeError('event loop failed to release transport')
            return 'result'
        async def run():
            with patch.object(client._client, 'system_one', side_effect=blocking):
                task = asyncio.create_task(client.system_one(state='', questions=questions()))
                for _ in range(100):
                    if started.is_set():
                        break
                    await asyncio.sleep(.01)
                self.assertTrue(started.is_set())
                release.set()
                self.assertEqual(await task, 'result')
        try:
            asyncio.run(run())
        finally:
            release.set()

    def test_grouped_empty_maps(self):
        with running() as (_, url):
            result = TypeSafeClient(base_url=url).system_one(state='', questions={'only': Noul(instructions='Yes?')})
            self.assertEqual(result.choices, {})
            self.assertEqual(result.scores, {})
            self.assertEqual(set(result.nouls), {'only'})
