# Multimodal preprocessing and provenance

This runtime ships the offline multimodal **preprocessing** path only. Training,
dataset generation, and the dataset converter are not part of this release and
are not shipped with it; the `jev_laya_free.trainer` module does not exist in
this repository.

## What is shipped

The opt-in preprocessing functions in `jev_laya_free.multimodal` return
deterministic, JSON-compatible feature records from in-memory text/bytes. They
are not registered in the request schema, client, backend, or server, and they
add no decision, explanation, or rationale. Full behavior is documented in
`src/jev_laya_free/multimodal/README.md`.

* `preprocess_code` and `preprocess_pdf` — no libraries required for text or
  code; optional `pypdf` for PDF extraction; optional Pillow for image
  inspection. Nothing installs or downloads dependencies automatically.
* `prepare_video` and `prepare_audio` — probe-gated and fail-closed: without a
  verified local probe they return `unavailable` (`video_not_verified` /
  `audio_not_verified`) without decoding.

No model, service, or network is required for any of these paths.

## Provenance boundaries

* **Local-only inputs.** Preprocessing consumes in-memory bytes supplied by the
  caller; raw media bytes never enter the record or its `repr`.
* **Bounded features.** Limits bound input bytes, selected pages, output
  text/snippets, AST nodes, and image pixels. They do not impose CPU deadlines
  on third-party parsers or trusted callbacks. For untrusted artifacts, the
  caller must isolate parsing and enforce process memory/time limits.
* **Redaction is best-effort.** Text and snippets are normalized, bounded, and
  have absolute path tokens redacted. Extracted text can still contain private
  content; these are local feature records, not an anonymization or
  secret-scanning system.
* **Capability states are explicit.** Records carry `kind`, `status` (`ok`,
  `partial`, `unavailable`), `features`, and `capabilities`. Unavailable
  artifacts have a fixed `reason` code; capability entries expose `available`
  and `reason`, never library exception messages.
* **No quality claim.** The preprocessing records are conformance evidence only.
  A capability claim requires held-out support from the separate training
  environment that passes the acceptance gates; endpoint presence alone never
  counts as verified.

## What is not shipped

* The dataset converter (the removed `jev_laya_free.trainer.conversion` module — training is not shipped) and its source policy for AgentHazard, ETO, Hermes, Sentinel, Pi sessions, and Visual/GUI sources.
* The `configs/dataset_sources.json` file remains in the repository as the
  recorded source policy, but no in-repo command consumes it.
* Train/validation/test JSONL generation, manifests, and calibration data.

Re-running the same preprocessing inputs with the same limits produces identical
records; the path is a pure function of its inputs.
