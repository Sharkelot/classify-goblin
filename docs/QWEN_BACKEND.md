# JEV-MM-12 / JEV-MM-13 — Local Qwen typed backend (image/PDF/code/video/audio)

A local-only, fail-closed artifact backend that converts verified
`ResolvedArtifact` bytes into a typed JSON decision via the loopback
OpenAI-compatible Qwen endpoint. It is wired through `predict_artifacts`
so text-only requests keep their existing behavior. JEV-MM-13 adds the
bounded video adapter (probe-gated, frame-sampled) and the fail-closed
audio adapter, and wires video into the verified message shape.

## Files

- **Backend + client:** `src/jev_laya_free/multimodal/qwen_service.py`
  - `OpenAICompatClient` — loopback-only OpenAI-compatible transport
    (explicit endpoint or `JEV_QWEN_BASE_URL`; credential from
    `JEV_QWEN_API_KEY`; bounded 1..120 s timeout).
  - `QwenArtifactBackend` — capability-gated, fail-closed; `predict` and
    `predict_artifacts` both route through `schema.answers` validation.
- **Server wiring:** `src/jev_laya_free/server.py` — `--backend qwen`
  builds a `QwenArtifactBackend` from `JEV_QWEN_BASE_URL` /
  `JEV_QWEN_TIMEOUT`; the handler already routes nonempty `artifacts`
  through `predict_artifacts` and returns 503 when the backend cannot
  serve them (no fallback to text).
- **Tests:** `tests/test_qwen_service.py` — fake/injected transport only
  (no live service), plus server-level wiring tests that run the real
  `LocalServer`.

## Modalities: verified vs. unavailable

The backend declares `SUPPORTED = {image, pdf, code, video}`. The JEV-MM-01
runtime probe (historical evidence, SHA
`31ed208a11d720d7aa06f0c87b74923318ed4f4cc41f4ac9a3cb6005e7997e3e`)
and the fresh JEV-MM-13 probe (2026-09-21) observed the following against
the live endpoint:

| Modality | Probe state | This card |
|----------|-------------|-----------|
| image    | verified    | supported |
| code     | (new)       | supported |
| pdf      | unavailable | supported (declared; probe did not verify) |
| video    | verified    | supported (JEV-MM-13: adapter + message shape) |
| audio    | unavailable | fail-closed adapter (capability `audio_not_verified`) |

`pdf` and `code` are the new capabilities JEV-MM-12 adds; `pdf` was not
verified by the historical probe, so it is declared supported by the
backend but its live verification is the responsibility of a future probe
run. **Video** is verified by the JEV-MM-13 probe (2026-09-21: a
`video_url` data URI returned HTTP 200 with `multimodal_tokens.video`
and a correct answer), so the backend now declares it supported and the
workflow routes `video` → `inspect_video`. **Audio** is not supported by
the endpoint (HTTP 400 "At most 0 audio(s) may be provided"; no
transcription route), so the audio adapter is fail-closed: it returns
`audio_not_verified` rather than a fake success. An explicit
`local_capabilities` override still wins at negotiation time. Unsupported
modalities (unknown) fail closed with a stable
`QwenCapabilityUnavailable` → 503, never a fallback to text.

## Video adapter (JEV-MM-13)

`prepare_video(artifact, *, probe_verified=False, frame_sampler=None,
limits=DEFAULT_LIMITS)` returns a `MediaPayload`. It is **probe-gated and
fail-closed**: unless `probe_verified` is true it returns
`status='unavailable', reason='video_not_verified'` without touching the
bytes. The default frame sampler uses `ffprobe` + `ffmpeg` and enforces:

| Limit | Value | Failure reason |
|-------|-------|----------------|
| Duration | ≤ 120 s | `video_too_long` |
| Sampled frame count | ≤ 16 (at 2 fps) | `too_many_frames` |
| Per-frame resolution | ≤ 4,096 × 2,160 | `frame_too_large` |
| Total bytes | ≤ 64 MiB | `invalid_or_oversize_bytes` |
| Container | MP4 / WebM | `unsupported_container` |
| Decode / probe failure | — | `video_decode_failed` |

Frames are sampled at 2 fps and each carries `(index, timestamp, width,
height, bytes)` plus a SHA-256 of the JPEG bytes; the raw video bytes are
not stored or logged. The Qwen backend sends the video as a `video_url`
data URI (the verified shape), not as individual frames.

## Audio adapter (JEV-MM-13)

`prepare_audio(artifact, *, probe_verified=False, limits=AudioLimits())`
returns a `MediaPayload`. It is **fail-closed**: unless `probe_verified`
is true it returns `status='unavailable', reason='audio_not_verified'`.
It gates on a MIME allow-list (`audio/wav`, `audio/x-wav`, `audio/mpeg`,
`audio/mp3`) — not magic bytes — and parses WAV headers to require PCM
(`audio_format == 1`); any other format returns
`unsupported_audio_format`. Because the local endpoint reports audio as
unavailable (limit 0, no transcription route), the adapter does not
attempt a transcript and never returns a fake success.

## Live probe result (reported separately from fake/injected tests)

Probed **2026-09-21** against the live loopback vLLM endpoint
`http://127.0.0.1:8080` (model `Qwen3.8`, `max_model_len` 200000):

- **Reachable:** yes. `/health` and `/v1/models` respond; the model id
  is `Qwen3.8`.
- **Image tokens accounted:** yes — a single-image request reported
  `multimodal_tokens.image = 64`.
- **Typed decision (image + code):** the endpoint returns a well-formed
  typed JSON decision, but the model's `score` (2) did not match its own
  probability weighted-mean (1.9), so strict `schema.answers` validation
  **fails closed** (502/503, no fallback). This is the honest live
  result: the endpoint is up and multimodal, but strict score validation
  is not guaranteed to pass for every prompt. A choice-only request
  through the full server (image artifact, `Route` question) returned
  **200** with sanitized usage.
- **No fabricated live pass:** the fake/injected tests are the
  deterministic contract; the live probe is reported as-is and does not
  substitute for them.

### JEV-MM-13 probe (2026-09-21, fresh)

Probed **2026-09-21** against the same live loopback endpoint
(`http://127.0.0.1:8080`, model `Qwen3.8`):

- **Video verified:** a `video_url` data URI (64×64, 1 s MP4) returned
  HTTP 200 with `multimodal_tokens.video = 24`; the model answered
  "Red" for a red clip. The endpoint consumes raw video, so the backend
  sends the whole clip as a `video_url` data URI.
- **Audio unavailable:** both `input_audio` and `audio_url` part types
  returned HTTP 400 ("At most 0 audio(s) may be provided") and
  `/v1/audio/transcriptions` returned 404. The audio adapter is
  therefore fail-closed (`audio_not_verified`); no transcript is
  claimed.
- **No auth header required** (loopback, vLLM default).

## JEV-MM-14 — capability/task identity in the prompt

The system prompt now makes the capability and (optional) task identity
explicit instead of embedding them only in the raw state:

- `Capability under evaluation: <modality>.` — the state's `modality`,
  or `text` when the state is not a dict or has no modality.
- `Task identity: <id>` — present only when the state carries a
  `task_identity` string (the workflow's deterministic per-state
  fingerprint).

These lines are **informational**: they label which capability is under
evaluation so the model can ground its typed answers, but they never
authorize anything. The typed answer schema and strict `schema.answers`
validation remain the only contract the model output must satisfy; a
non-conforming answer still fails closed (502/503, no fallback).

## Config boundary

- Endpoint: explicit `base_url` or `JEV_QWEN_BASE_URL`
  (default `http://127.0.0.1:8080`); loopback-only (`127.0.0.1` /
  `localhost`), no userinfo, no path.
- Credential: `JEV_QWEN_API_KEY` (env); never embedded in the URL or
  model id, never printed.
- Model: `JEV_QWEN_MODEL` (default `Qwen3.8`).
- Timeout: bounded 1..120 s (`JEV_QWEN_TIMEOUT` for the server entry
  point).
- No filesystem paths or arbitrary URLs are sent; artifacts are
  base64-encoded in-memory bytes.
- Usage is sanitized to artifact id / pages / truncation plus standard
  token counts — no raw bytes, no local path, no filename, no provider
  error body.
- No model output authorizes workflow actions or delegation; it is a
  typed decision only.
