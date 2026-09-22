"""Integration tests for the reference service and the independent client.

These tests start the reference service on a random loopback port and
exercise the full request/response cycle, including error paths and
transport constraints.
"""
import json
import socket
import threading
import unittest
from classify_goblin import protocol
from classify_goblin.protocol_client import ProtocolClient, ProtocolClientError
from classify_goblin.reference_service import ReferenceServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServiceClientBase(unittest.TestCase):
    def setUp(self):
        self.port = _free_port()
        self.server = ReferenceServer(("127.0.0.1", self.port))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def make_client(self, **kwargs):
        kwargs.setdefault("base_url", self.base_url)
        kwargs.setdefault("retries", 0)
        return ProtocolClient(**kwargs)

    def basic_questions(self):
        return {
            "noul_q": {"type": "noul", "instructions": "Should we proceed?"},
            "choice_q": {"type": "choice", "instructions": "Which tier?",
                         "options": ["code", "llm"]},
            "score_q": {"type": "score", "instructions": "How confident?",
                         "criteria": ["low", "high"]},
        }


class HappyPathTests(ServiceClientBase):
    def test_round_trip(self):
        client = self.make_client()
        result = client.system_one(state={"modality": "code"},
                                   questions=self.basic_questions())
        self.assertEqual(result["model"], "local-rules-v1")
        self.assertIn("request_id", result)
        self.assertEqual(result["usage"]["input_tokens"], 0)
        self.assertEqual(result["usage"]["output_tokens"], 0)
        self.assertEqual(set(result["answers"]), set(self.basic_questions()))

    def test_answers_are_validated(self):
        client = self.make_client()
        result = client.system_one(state={"modality": "code"},
                                   questions=self.basic_questions())
        # noul answer
        self.assertIn("noul", result["answers"]["noul_q"])
        self.assertIsInstance(result["answers"]["noul_q"]["noul"], float)
        # choice answer
        self.assertIn("choice", result["answers"]["choice_q"])
        self.assertIn("probabilities", result["answers"]["choice_q"])
        # score answer
        self.assertIn("score", result["answers"]["score_q"])
        self.assertIn("legend", result["answers"]["score_q"])

    def test_client_context_manager(self):
        with self.make_client() as client:
            result = client.system_one(state={}, questions=self.basic_questions())
            self.assertIn("answers", result)


class ErrorPathTests(ServiceClientBase):
    def test_404_wrong_path(self):
        import urllib.request
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/wrong",
            data=b"{}", method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            self.fail("expected HTTP error")
        except Exception as exc:
            self.assertEqual(getattr(exc, "code", None), 404)

    def test_415_wrong_content_type(self):
        import urllib.request
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{protocol.ENDPOINT}",
            data=b"{}", method="POST",
            headers={"Content-Type": "text/plain"},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            self.fail("expected HTTP error")
        except Exception as exc:
            self.assertEqual(getattr(exc, "code", None), 415)

    def test_401_bad_token(self):
        port = _free_port()
        server = ReferenceServer(("127.0.0.1", port), token="secret")
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            client = ProtocolClient(base_url=f"http://127.0.0.1:{port}",
                                    api_key="wrong", retries=0)
            with self.assertRaises(ProtocolClientError) as ctx:
                client.system_one(state={}, questions=self.basic_questions())
            self.assertEqual(ctx.exception.status, 401)
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_401_correct_token(self):
        port = _free_port()
        server = ReferenceServer(("127.0.0.1", port), token="secret")
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            client = ProtocolClient(base_url=f"http://127.0.0.1:{port}",
                                    api_key="secret", retries=0)
            result = client.system_one(state={}, questions=self.basic_questions())
            self.assertIn("answers", result)
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=5)

    def test_422_invalid_request(self):
        client = self.make_client()
        with self.assertRaises(ProtocolClientError):
            client.system_one(state=42, questions=self.basic_questions())

    def test_422_unknown_model(self):
        client = self.make_client()
        with self.assertRaises(ProtocolClientError) as ctx:
            client.system_one(state={}, questions=self.basic_questions(),
                             model="nonexistent-model")
        self.assertEqual(ctx.exception.status, 422)

    def test_422_bad_question_server_side(self):
        import urllib.request
        body = {"model": "local-rules-v1", "state": {},
                "questions": {"q": {"type": "boolean", "instructions": "Q?"}}}
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{protocol.ENDPOINT}",
            data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            self.fail("expected HTTP error")
        except Exception as exc:
            self.assertEqual(getattr(exc, "code", None), 422)

    def test_client_side_validation_wraps_error(self):
        client = self.make_client()
        with self.assertRaises(ProtocolClientError) as ctx:
            client.system_one(state=42, questions=self.basic_questions())
        # client-side validation error has no HTTP status
        self.assertIsNone(ctx.exception.status)


class TransportConstraintTests(ServiceClientBase):
    def test_rejects_non_loopback_url(self):
        with self.assertRaises(ValueError):
            ProtocolClient(base_url="http://example.com:8093")

    def test_rejects_non_http_url(self):
        with self.assertRaises(ValueError):
            ProtocolClient(base_url="https://127.0.0.1:8093")

    def test_rejects_bad_timeout(self):
        with self.assertRaises(ValueError):
            ProtocolClient(base_url=self.base_url, timeout=0)

    def test_rejects_bad_retries(self):
        with self.assertRaises(ValueError):
            ProtocolClient(base_url=self.base_url, retries=6)

    def test_rejects_bad_api_key(self):
        with self.assertRaises(ValueError):
            ProtocolClient(base_url=self.base_url, api_key="")


class HealthEndpointTests(ServiceClientBase):
    def test_health(self):
        import urllib.request
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}/health", timeout=5
        ) as resp:
            data = json.loads(resp.read())
            self.assertEqual(data["status"], "ok")


if __name__ == "__main__":
    unittest.main()
