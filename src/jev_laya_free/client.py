"""Independent client; this module is not the official typesafe_sdk package."""
import asyncio
import os
import time
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler
from . import schema


class ClientError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class Record(dict):
    """JSON mapping with convenience attribute access; use [] for colliding names."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def record(value):
    if isinstance(value, dict):
        return Record({k: record(v) for k, v in value.items()})
    if isinstance(value, list):
        return [record(v) for v in value]
    return value


def Choice(*, instructions, options=None, criteria=None):
    schema.require((options is None) != (criteria is None), 'provide options or criteria')
    return {'type': 'choice', 'instructions': instructions,
            **({'options': options} if options is not None else {'criteria': criteria})}


def Score(*, instructions, criteria):
    return {'type': 'score', 'instructions': instructions, 'criteria': criteria}


def Noul(*, instructions, criteria=None):
    return {'type': 'noul', 'instructions': instructions, **({'criteria': criteria} if criteria is not None else {})}


class Response(Record):
    """Wire mapping plus grouped SDK views; views do not add wire fields."""
    def _group(self, kind):
        return Record({name: answer for name, answer in self['answers'].items()
                       if answer['type'] == kind})

    @property
    def choices(self):
        return self._group('choice')

    @property
    def scores(self):
        return self._group('score')

    @property
    def nouls(self):
        return self._group('noul')


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class TypeSafeClient:
    """Familiar method names, independent implementation and localhost-only transport."""
    def __init__(self, base_url=None, api_key=None,
                 model='local-default', timeout=5.0, retries=2):
        base_url = base_url if base_url is not None else os.environ.get('TYPESAFE_BASE_URL', 'http://127.0.0.1:8093')
        parsed = urlsplit(base_url)
        schema.require(parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost')
                       and parsed.path in ('', '/') and not parsed.query and not parsed.fragment
                       and parsed.username is None and parsed.password is None,
                       'base_url must be a loopback HTTP origin')
        schema.require(type(timeout) in (int, float) and 0 < timeout <= 60, 'timeout must be 0..60 seconds')
        schema.require(type(retries) is int and 0 <= retries <= 5, 'retries must be 0..5')
        self.url = base_url.rstrip('/') + '/v1/systemone'
        self.api_key = api_key if api_key is not None else os.environ.get('JEV_LOCAL_API_KEY', os.environ.get('TYPESAFE_API_KEY'))
        schema.require(self.api_key is None or (isinstance(self.api_key, str) and self.api_key.isascii()
                       and all(33 <= ord(c) <= 126 for c in self.api_key) and bool(self.api_key)), 'invalid api_key')
        self.model, self.timeout, self.retries = model, timeout, retries
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def system_one(self, *, state, questions, model=None, artifacts=None):
        body = schema.request({'state': state, 'questions': questions, 'model': self.model if model is None else model,
                               **({'artifacts': artifacts} if artifacts is not None else {})})
        encoded = schema.dumps(body).encode()
        headers = {'Content-Type': 'application/json'}
        if self.api_key is not None:
            headers['Authorization'] = 'Bearer ' + self.api_key
        for attempt in range(self.retries + 1):
            try:
                with self.opener.open(Request(self.url, data=encoded, headers=headers, method='POST'), timeout=self.timeout) as reply:
                    schema.require(reply.headers.get_content_type() == 'application/json', 'invalid response content type')
                    data = reply.read(schema.MAX_RESPONSE_BYTES + 1)
                    schema.require(len(data) <= schema.MAX_RESPONSE_BYTES, 'response too large')
                return Response(record(schema.response(schema.loads(data), body['questions'])))
            except HTTPError as exc:
                status = exc.code
                exc.close()
                if status not in (429, 502, 503, 504, 529) or attempt == self.retries:
                    raise ClientError(f'local HTTP error {status}', status) from exc
            except (URLError, TimeoutError, OSError, HTTPException) as exc:
                if attempt == self.retries:
                    raise ClientError('local transport failed') from exc
            except (schema.ValidationError, ValueError, RecursionError, UnicodeError) as exc:
                raise ClientError('invalid decision response; no fallback answer produced') from exc
            time.sleep(min(0.1 * 2 ** attempt, 2.0))
        raise AssertionError('unreachable')

    systemOne = system_one

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class AsyncTypeSafeClient:
    """Async facade over the same bounded local transport, off the event loop.

    Cancellation stops awaiting but cannot interrupt the worker's socket operation.
    """
    def __init__(self, *args, **kwargs):
        self._client = TypeSafeClient(*args, **kwargs)

    async def system_one(self, *, state, questions, model=None, artifacts=None):
        return await asyncio.to_thread(self._client.system_one,
                                       state=state, questions=questions, model=model, artifacts=artifacts)

    systemOne = system_one

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False
