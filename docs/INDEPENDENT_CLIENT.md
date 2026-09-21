# Independent Client for the Jev-Compatible Protocol

This is a minimal, dependency-free Python client for the typed-decision
protocol described in [PROTOCOL.md](./PROTOCOL.md). It is written
against `protocol.py` only — it does **not** import from
`reference_service`, so it can be dropped into a third-party codebase
and verified against any conforming server.

**Dependencies.** Standard library only: `urllib`, `json`, `math`,
`uuid`, `time`, `http.client`, `urllib.error`, `urllib.parse`. No
third-party packages.

## Quick start

```python
from jev_laya_free.protocol_client import ProtocolClient, ProtocolClientError

client = ProtocolClient(base_url="http://127.0.0.1:8093", retries=2)

questions = {
    "model_routing": {
        "type": "choice",
        "instructions": "Which execution tier fits this turn?",
        "options": ["deterministic-code", "cheap-LLM", "frontier-LLM", "human"],
    },
    "completion": {"type": "noul", "instructions": "Should completion be held for review?"},
}

result = client.system_one(state={"modality": "code"}, questions=questions)
print(result["answers"]["model_routing"]["choice"])  # e.g. "deterministic-code"
```

With bearer-token auth:

```python
client = ProtocolClient(base_url="http://127.0.0.1:8093", api_key="secret")
```

## API

### `ProtocolClient`

| Parameter | Default | Description |
|---|---|---|
| `base_url` | `JEV_PROTOCOL_BASE_URL` env or `http://127.0.0.1:8093` | Loopback HTTP origin. Must be `http`, host `127.0.0.1`/`localhost`, path empty or `/`, no query/fragment/credentials. |
| `api_key` | `JEV_LOCAL_API_KEY` / `TYPESAFE_API_KEY` env | Optional bearer token. ASCII, printable, non-empty. |
| `model` | `"local-default"` | Default model name sent when `system_one` is called without `model`. |
| `timeout` | `5.0` | Per-request timeout in seconds. 0 < timeout ≤ 60. |
| `retries` | `2` | Retry count on `429, 502, 503, 504, 529`. 0..5. |

The constructor validates the base URL, timeout, retries, and api_key.
It raises `ValueError` on any constraint violation.

### `system_one(*, state, questions, model=None)`

Send a typed-decision request and return the validated response dict.

- `state`: any JSON value (string, dict, or list).
- `questions`: mapping of question name → question object.
- `model`: override the default model for this request.

Returns a dict with `model`, `answers`, `usage`, and `request_id`.
The response is validated with `protocol.validate_response` before
returning.

Raises `ProtocolClientError` on transport failure, schema failure, or an
unretryable HTTP error.

### Context manager

```python
with ProtocolClient(base_url="http://127.0.0.1:8093") as client:
    result = client.system_one(state={}, questions={...})
```

The context manager is a no-op (the client holds no resources that need
closing), but it is provided for API symmetry with the existing
`TypeSafeClient`.

## Error handling

`ProtocolClientError` carries an optional `status` attribute:

- `status=None` — client-side validation failure (bad base URL, bad
  request schema, invalid response schema).
- `status=<int>` — server returned a non-retryable HTTP error.

```python
from jev_laya_free.protocol_client import ProtocolClient, ProtocolClientError

try:
    client.system_one(state=42, questions={...})  # 42 is not a valid state
except ProtocolClientError as e:
    print(e.status)  # None — client-side validation
    print(str(e))    # "invalid request: state must be string, object, or array"
```

## Verifying compatibility

To verify that a third-party client is compatible with the reference
service:

1. Start the reference service:
   ```bash
   PYTHONPATH=src python -m jev_laya_free.reference_service --port 8093
   ```
2. Run the compatibility tests:
   ```bash
   PYTHONPATH=src python -m pytest tests/test_reference_service.py -v
   ```

The tests cover:
- Happy-path round-trip (all three question types)
- Answer schema validation (noul, choice, score)
- Error paths (401, 404, 413, 415, 422, 503, 529)
- Retry behavior on 503
- Transport constraints (non-loopback URL rejection, non-ASCII api_key rejection)
- Context manager
- Health endpoint

## Differences from the existing `TypeSafeClient`

| Property | `TypeSafeClient` | `ProtocolClient` |
|---|---|---|
| Imports | `schema.py` (shared with server) | `protocol.py` only |
| Response wrapper | `Response` (Record subclass with `.choices`, `.scores`, `.nouls`) | plain `dict` |
| Error type | `ClientError` | `ProtocolClientError` |
| Retryable statuses | `429, 502, 503, 504, 529` | same |
| Transport | `urllib` (no proxy, no redirect) | same |
| Loopback check | yes | yes |

`ProtocolClient` is intentionally simpler: it returns a plain dict and
has no convenience views. If you need `.choices` / `.scores` / `.nouls`
grouping, use `TypeSafeClient` or add the views yourself.
