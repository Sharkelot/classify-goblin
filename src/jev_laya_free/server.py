"""Small loopback HTTP service; no hosted inference or document/tool execution."""
import argparse
import hmac
import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from . import schema
from .backends import LayaBackend, RulesBackend


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(self, address, backend, token=None):
        schema.require(address[0] in ('127.0.0.1', 'localhost'), 'loopback binding required')
        schema.require(token is None or (isinstance(token, str) and token.isascii() and all(33 <= ord(c) <= 126 for c in token) and bool(token)), 'invalid bearer token')
        self.backend, self.token = backend, token
        self.inference_lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(16)
        super().__init__(address, Handler)

    def process_request(self, request, client_address):
        # Bound active request threads, including clients that send headers slowly.
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *args):
        pass  # No state, credentials, or question content in access logs.

    def send_json(self, status, payload):
        encoded = schema.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(encoded)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self):
        request_id = str(uuid.uuid4())
        def error(code, message):
            self.send_json(code, {'error': {'code': code, 'message': message}, 'request_id': request_id})
        if self.path != '/v1/systemone':
            error(404, 'not found')
            return
        if self.server.token is not None:
            provided = self.headers.get('Authorization', '')
            if not hmac.compare_digest(provided.encode(), ('Bearer ' + self.server.token).encode()):
                error(401, 'invalid bearer token')
                return
        if self.headers.get_content_type() != 'application/json':
            error(415, 'Content-Type must be application/json')
            return
        lengths = self.headers.get_all('Content-Length', [])
        if self.headers.get('Transfer-Encoding') or len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit() or len(lengths[0]) > 10:
            error(400, 'one Content-Length required; chunked bodies unsupported')
            return
        length = int(lengths[0])
        if not 1 <= length <= schema.MAX_BYTES:
            error(413, 'request body must be 1..65536 bytes')
            return
        try:
            data = self.rfile.read(length)
            schema.require(len(data) == length, 'incomplete body')
            body = schema.request(schema.loads(data))
            schema.require(body['model'] in ('local-default', self.server.backend.model), 'unknown local model')
        except (schema.ValidationError, ValueError, RecursionError, UnicodeError):
            error(422, 'invalid request schema or local model')
            return
        except (TimeoutError, OSError):
            error(408, 'request read timed out')
            return
        if not self.server.inference_lock.acquire(blocking=False):
            error(529, 'local backend busy')
            return
        try:
            raw = self.server.backend.predict(body['state'], body['questions'])
            result = {'model': self.server.backend.model,
                      'answers': schema.answers(raw.get('answers'), body['questions']),
                      'usage': raw.get('usage'), 'request_id': request_id}
            schema.response(result, body['questions'])
        except schema.ValidationError:
            error(502, 'backend rejected request or returned invalid output')
            return
        except Exception:
            error(503, 'local backend unavailable')
            return
        finally:
            self.server.inference_lock.release()
        self.send_json(200, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=('127.0.0.1', 'localhost'), default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8093)
    parser.add_argument('--backend', choices=('rules', 'laya'), default='rules')
    args = parser.parse_args()
    backend = LayaBackend() if args.backend == 'laya' else RulesBackend()
    server = LocalServer((args.host, args.port), backend, os.environ.get('JEV_LOCAL_API_KEY'))
    print(f'Local typed decisions: http://{args.host}:{server.server_port}/v1/systemone ({backend.model})', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
