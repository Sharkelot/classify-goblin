"""Reference implementation of the Jev-compatible typed-decision protocol.

This module provides a minimal, self-contained HTTP server that implements
the wire contract defined in ``protocol.py``.  It uses only the Python
standard library.  The server is intentionally simple: one POST endpoint,
bounded JSON bodies, and a deterministic rules backend.

Run directly::

    PYTHONPATH=src python -m jev_laya_free.reference_service --port 8093
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from . import protocol


# ---------------------------------------------------------------------------
# Deterministic rules backend (same as the existing RulesBackend)
# ---------------------------------------------------------------------------


def _confidence(probabilities: Dict[str, float]) -> float:
    values = list(probabilities.values())
    if len(values) == 1:
        return 1.0
    return max(0.0, 1 + sum(p * math.log(p) for p in values if p) / math.log(len(values)))


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def predict(state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministic lexical-overlap prediction (NOT a semantic classifier)."""
    tokens = set(re.findall(r"\w+", _text(state).lower()))

    def distribution(labels: Dict[str, str]) -> Dict[str, float]:
        weights = {k: 1 + len(tokens & set(re.findall(r"\w+", _text(v).lower())))
                   for k, v in labels.items()}
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    output: Dict[str, Any] = {}
    for name, q in questions.items():
        kind = q["type"]
        if kind == "noul":
            criteria = q.get("criteria", {})
            p = distribution({k: criteria.get(k, "") for k in ("false", "true")})
            output[name] = {"type": kind, "noul": p["true"]}
            continue
        labels = (
            {k: k + " " + (_text(v) if v is not None else "") for k, v in q["criteria"].items()}
            if kind == "choice"
            else {str(i): _text(v) for i, v in enumerate(q["criteria"])}
        )
        p = distribution(labels)
        a: Dict[str, Any] = {"type": kind, "probabilities": p, "confidence": _confidence(p)}
        if kind == "choice":
            a["choice"] = max(p, key=p.get)
        else:
            a["score"] = sum(int(k) * v for k, v in p.items())
            a["legend"] = labels
        output[name] = a
    return {"answers": output, "usage": {"input_tokens": 0, "output_tokens": 0}}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class ReferenceHandler(BaseHTTPRequestHandler):
    """Handles a single request against the protocol spec."""

    def log_message(self, *args):
        pass  # No state, credentials, or question content in access logs.

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                            separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", protocol.CONTENT_TYPE)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _error(self, code: int, message: str, request_id: str) -> None:
        self._send_json(code, {"error": {"code": code, "message": message},
                              "request_id": request_id})

    def do_POST(self):
        request_id = str(uuid.uuid4())

        # --- path check ---
        if self.path != protocol.ENDPOINT:
            self._error(protocol.STATUS_NOT_FOUND, "not found", request_id)
            return

        # --- auth check ---
        token = self.server.token
        if token is not None:
            provided = self.headers.get("Authorization", "")
            if not hmac.compare_digest(provided.encode(), ("Bearer " + token).encode()):
                self._error(protocol.STATUS_UNAUTHORIZED, "invalid bearer token", request_id)
                return

        # --- content-type check ---
        if self.headers.get_content_type() != protocol.CONTENT_TYPE:
            self._error(protocol.STATUS_UNSUPPORTED_MEDIA,
                        f"Content-Type must be {protocol.CONTENT_TYPE}", request_id)
            return

        # --- content-length check ---
        if self.headers.get(protocol.CHUNKED_TE_HEADER):
            self._error(protocol.STATUS_BAD_REQUEST,
                        "chunked transfer encoding not supported", request_id)
            return
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit() \
                or len(lengths[0]) > protocol.MAX_CONTENT_LENGTH_DIGITS:
            self._error(protocol.STATUS_BAD_REQUEST,
                        "one Content-Length required; chunked bodies unsupported", request_id)
            return
        length = int(lengths[0])
        if not (1 <= length <= protocol.MAX_REQUEST_BYTES):
            self._error(protocol.STATUS_REQUEST_TOO_LARGE,
                        f"request body must be 1..{protocol.MAX_REQUEST_BYTES} bytes",
                        request_id)
            return

        # --- read body ---
        try:
            data = self.rfile.read(length)
            if len(data) != length:
                self._error(protocol.STATUS_BAD_REQUEST, "incomplete body", request_id)
                return
            body = json.loads(data)
        except (ValueError, UnicodeError):
            self._error(protocol.STATUS_UNPROCESSABLE, "invalid JSON", request_id)
            return
        except (TimeoutError, OSError):
            self._error(protocol.STATUS_REQUEST_TIMEOUT, "request read timed out", request_id)
            return

        # --- validate request schema ---
        try:
            protocol.validate_request(body)
        except ValueError as exc:
            self._error(protocol.STATUS_UNPROCESSABLE, str(exc), request_id)
            return

        # --- model check ---
        if body["model"] not in ("local-default", self.server.backend_model):
            self._error(protocol.STATUS_UNPROCESSABLE, "unknown local model", request_id)
            return

        # --- inference lock ---
        if not self.server.inference_lock.acquire(blocking=False):
            self._error(protocol.STATUS_BACKEND_BUSY, "local backend busy", request_id)
            return

        try:
            normalized = protocol.normalize_questions(body["questions"])
            raw = predict(body["state"], normalized)
            result = {
                "model": self.server.backend_model,
                "answers": raw["answers"],
                "usage": raw["usage"],
                "request_id": request_id,
            }
            protocol.validate_response(result, normalized)
        except ValueError as exc:
            self._error(protocol.STATUS_BAD_GATEWAY,
                        f"backend rejected request or returned invalid output: {exc}",
                        request_id)
            return
        except Exception:
            self._error(protocol.STATUS_SERVICE_UNAVAILABLE, "local backend unavailable",
                        request_id)
            return
        finally:
            self.server.inference_lock.release()

        self._send_json(protocol.STATUS_OK, result)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._error(protocol.STATUS_NOT_FOUND, "not found", str(uuid.uuid4()))


class ReferenceServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(self, address: tuple, token: Optional[str] = None,
                 backend_model: str = "local-rules-v1"):
        if address[0] not in protocol.ALLOWED_HOSTNAMES:
            raise ValueError("loopback binding required")
        if token is not None:
            if not (isinstance(token, str) and token.isascii()
                    and all(33 <= ord(c) <= 126 for c in token) and bool(token)):
                raise ValueError("invalid bearer token")
        self.token = token
        self.backend_model = backend_model
        self.inference_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(16)
        super().__init__(address, ReferenceHandler)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("127.0.0.1", "localhost"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()
    server = ReferenceServer((args.host, args.port), args.token)
    print(f"Reference typed decisions: http://{args.host}:{server.server_port}"
          f"{protocol.ENDPOINT} ({server.backend_model})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
