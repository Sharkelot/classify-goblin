"""Independent client for the Benchmark Heaven-compatible typed-decision protocol.

This client is written against ``protocol.py`` only — it does NOT import
from ``reference_service``.  It uses only the Python standard library.

The client enforces the same transport rules as the spec: loopback HTTP
origin only, one Content-Length, no chunked transfer, bounded bodies, and
retry on the spec's retryable status codes.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from . import protocol


class ProtocolClientError(RuntimeError):
    """Raised when a request fails at the transport or schema level."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ProtocolClient:
    """Minimal client for the Benchmark Heaven-compatible typed-decision protocol.

    Parameters
    ----------
    base_url:
        Loopback HTTP origin, e.g. ``http://127.0.0.1:8093``.
    api_key:
        Optional bearer token.
    model:
        Model identifier to send in each request.
    timeout:
        Per-request timeout in seconds (0 < timeout <= 60).
    retries:
        Number of retries on retryable status codes (0 <= retries <= 5).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "local-default",
        timeout: float = 5.0,
        retries: int = 2,
    ):
        base_url = (
            base_url
            if base_url is not None
            else "http://127.0.0.1:8093"
        )
        protocol.validate_base_url(base_url)
        if not (isinstance(timeout, (int, float)) and 0 < timeout <= 60):
            raise ValueError("timeout must be 0..60 seconds")
        if not (isinstance(retries, int) and 0 <= retries <= 5):
            raise ValueError("retries must be 0..5")
        self.url = base_url.rstrip("/") + protocol.ENDPOINT
        self.api_key = api_key
        if self.api_key is not None:
            if not (isinstance(self.api_key, str) and self.api_key.isascii()
                    and all(33 <= ord(c) <= 126 for c in self.api_key)
                    and bool(self.api_key)):
                raise ValueError("invalid api_key")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self._opener = build_opener(ProxyHandler({}), NoRedirect())

    def system_one(
        self,
        *,
        state: Any,
        questions: Dict[str, Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a typed-decision request and return the validated response.

        Raises ``ProtocolClientError`` on transport or schema failure.
        """
        body = {
            "state": state,
            "questions": questions,
            "model": self.model if model is None else model,
        }
        try:
            protocol.validate_request(body)
        except ValueError as exc:
            raise ProtocolClientError(f"invalid request: {exc}") from exc
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False,
                            separators=(",", ":")).encode()
        headers = {"Content-Type": protocol.CONTENT_TYPE}
        if self.api_key is not None:
            headers["Authorization"] = "Bearer " + self.api_key

        for attempt in range(self.retries + 1):
            try:
                with self._opener.open(
                    Request(self.url, data=encoded, headers=headers,
                           method=protocol.HTTP_METHOD),
                    timeout=self.timeout,
                ) as reply:
                    if reply.headers.get_content_type() != protocol.CONTENT_TYPE:
                        raise ProtocolClientError("invalid response content type")
                    data = reply.read(protocol.MAX_RESPONSE_BYTES + 1)
                    if len(data) > protocol.MAX_RESPONSE_BYTES:
                        raise ProtocolClientError("response too large")
                result = json.loads(data)
                normalized = protocol.normalize_questions(body["questions"])
                protocol.validate_response(result, normalized)
                return result
            except HTTPError as exc:
                status = exc.code
                exc.close()
                if status not in protocol.RETRYABLE_STATUS_CODES or attempt == self.retries:
                    raise ProtocolClientError(f"local HTTP error {status}", status) from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt == self.retries:
                    raise ProtocolClientError("local transport failed") from exc
            except (ValueError, RecursionError, UnicodeError) as exc:
                raise ProtocolClientError(
                    "invalid decision response; no fallback answer produced"
                ) from exc
            time.sleep(min(0.1 * 2 ** attempt, 2.0))
        raise AssertionError("unreachable")

    systemOne = system_one

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
