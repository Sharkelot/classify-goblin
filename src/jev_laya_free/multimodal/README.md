# Local artifact preprocessing

These opt-in functions return deterministic JSON-compatible dictionaries; they
are not registered in the existing request schema, client, backend, or server.
They accept in-memory text/bytes rather than paths. No libraries are required for
text or code. PDF extraction uses optional `pypdf`; image inspection uses optional
Pillow. Nothing installs or downloads dependencies automatically.

```python
from jev_laya_free.multimodal import Limits, preprocess_code, preprocess_pdf

record = preprocess_code('def answer():\n    return 42', extension='.py')
record = preprocess_pdf(pdf_bytes, selected_pages=[1, 3], expected_page_count=4,
                        limits=Limits(max_text_chars=4096))
```

Records contain `kind`, `status` (`ok`, `partial`, `unavailable`), `features`, and
`capabilities`. Unavailable artifacts have a fixed `reason` code. Capability
entries expose `available` and `reason`, never library exception messages.
PDF page capabilities use compact outcome strings. `partial` can mean an
optional visual/AST capability is absent; callers should inspect the capability
they actually need. No decision, generated explanation, or rationale is added.

PDF page numbers are one-based, deduplicated and sorted. The default is the first
bounded set of pages. The document page limit and optional expected count are
validated before extraction; explicit out-of-range selections fail. The text
budget is shared across all selected pages. Metadata is structural: PDF version
and presence of standard fields; author/title contents are intentionally omitted.
If `pypdf` is absent, only an explicit selection can proceed through callbacks,
and page-count capability remains unavailable (including expected-count checks).
Encrypted and malformed PDFs return fixed errors.

Optional trusted adapters:

- `render_page(pdf_bytes, page_number)` returns a transient visual object.
- `ocr_page(pdf_bytes, page_number, visual_or_none)` returns text, used only when
  extraction is unavailable or empty. It may do its own rendering.
- `visual_input(pillow_image)` consumes decoded image pixels while the image is
  open. Its return value is ignored. Animated images use the first frame.

Visual objects, image metadata, callback return payloads and original artifact
bytes never enter records. Text and snippets are normalized, bounded and have
absolute path tokens redacted. Extracted text can still contain private content;
these are local feature records, not an anonymization or secret-scanning system.
Python code uses AST structure and locations (one-based lines and zero-based
UTF-8 byte columns); other languages receive bounded line snippets and language,
diff, caller-supplied test and lint signals. Test/lint tools are never executed.

Limits bound input bytes, selected pages, output text/snippets, AST nodes and
image pixels. They do not impose CPU deadlines on third-party parsers or trusted
callbacks. For untrusted artifacts, the caller must isolate parsing and enforce
process memory/time limits. Invalid limit configuration raises `ValueError` at
construction; artifact/parser/hook failures return explicit unavailable reasons.

## Video and audio (JEV-MM-13)

`prepare_video` and `prepare_audio` are probe-gated and fail-closed: without a
verified probe they return `unavailable` (`video_not_verified` /
`audio_not_verified`) without decoding. Video is frame-sampled at 2 fps with a
16-frame budget, a 120 s duration limit, and a 4,096 × 2,160 per-frame
resolution limit (see `docs/QWEN_BACKEND.md` for the full limits and failure
codes). Audio gates on a MIME allow-list (WAV/MP3), parses WAV headers to
require PCM, and never claims a transcript when the endpoint reports audio as
unavailable. Both return JSON-compatible `MediaPayload` records; raw media bytes
never enter the record or its `repr`.
