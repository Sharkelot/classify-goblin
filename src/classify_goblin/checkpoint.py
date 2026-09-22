"""Safe, dependency-free checkpoint downloader and cache installer.

The end-user path is:

    install classify-goblin runtime  ->  classify-goblin download-checkpoint  ->  run local inference

This module never embeds credentials, private URLs, or mutable ``latest``
references. The transport is injected (so tests use a local synthetic archive and
never download model weights during CI), the URL is policy-checked (HTTPS or
loopback HTTP only), the download is size-bounded and checksum-verified, the
archive is extracted with path/symlink traversal protection, and the install is
atomic (extract to a temp dir, then rename into place). Every failure path fails
closed with a specific exception; nothing pretends success.

The manifest is a small versioned JSON document (not the weights) carrying a
verified release artifact URL, exact SHA-256, size/format, expected files, and
compatibility version.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

# --- Exception surface (fail-closed, one per distinct failure mode) ---


class ManifestError(ValueError):
    """The manifest is malformed, or an expected file is missing after extract."""


class NoCanonicalArtifactError(RuntimeError):
    """No canonical public artifact URL is available; the downloader fails closed."""


class PolicyError(RuntimeError):
    """The artifact URL is not on the allowed transport policy (HTTPS/loopback)."""


class SizeLimitError(RuntimeError):
    """The download exceeded the bounded size limit."""


class PartialDownloadError(RuntimeError):
    """The transport returned fewer bytes than the declared artifact size."""


class ChecksumError(RuntimeError):
    """The downloaded bytes do not match the manifest SHA-256."""


class UnsafePathError(RuntimeError):
    """The archive contains a path/symlink that would escape the install dir."""


# --- Manifest validation ---

_REQUIRED_KEYS = ("version", "url", "sha256", "size", "format", "expected_files")
_ALLOWED_FORMATS = {"zip", "tar"}


def _is_relative(name: str) -> bool:
    return not name.startswith("/") and not name.startswith("\\")


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a versioned checkpoint manifest.

    Fails closed (``ManifestError``) on any missing/empty required key, a
    non-integer size, an unsupported format, or an absolute expected-file path.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"unreadable or invalid manifest: {path}") from exc
    if not isinstance(raw, dict):
        raise ManifestError("manifest must be a JSON object")
    for key in _REQUIRED_KEYS:
        if key not in raw:
            raise ManifestError(f"manifest missing required key: {key}")
    if not isinstance(raw["version"], str) or not raw["version"]:
        raise ManifestError("manifest version must be a non-empty string")
    if not isinstance(raw["url"], str):
        raise ManifestError("manifest url must be a string")
    sha = raw["sha256"]
    if not isinstance(sha, str) or len(sha) != 64:
        raise ManifestError("manifest sha256 must be a 64-char hex string")
    size = raw["size"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ManifestError("manifest size must be a non-negative integer")
    fmt = raw["format"]
    if fmt not in _ALLOWED_FORMATS:
        raise ManifestError(f"manifest format must be one of {sorted(_ALLOWED_FORMATS)}")
    expected = raw["expected_files"]
    if not isinstance(expected, list) or not expected:
        raise ManifestError("manifest expected_files must be a non-empty list")
    for name in expected:
        if not isinstance(name, str) or not _is_relative(name):
            raise ManifestError(f"manifest expected_files entry must be relative: {name!r}")
    return raw


def resolve_artifact(manifest: Mapping[str, Any]) -> str:
    """Return the canonical artifact URL, or fail closed if none is available."""
    url = manifest.get("url") or ""
    if not isinstance(url, str) or not url.strip():
        raise NoCanonicalArtifactError(
            "no canonical public checkpoint artifact URL is available; "
            "the downloader fails closed rather than guessing a mutable 'latest' reference"
        )
    return url


# --- Checksum ---


def verify_checksum(data: bytes, expected_sha256: str) -> None:
    """Fail closed (``ChecksumError``) unless the bytes match the expected SHA-256."""
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ChecksumError(f"checksum mismatch: expected {expected_sha256}, got {actual}")


# --- URL policy ---


def _check_policy(url: str) -> None:
    """Allow HTTPS, or HTTP restricted to loopback (127.0.0.1 / localhost)."""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    if scheme == "https":
        return
    if scheme == "http" and host in {"127.0.0.1", "localhost"}:
        return
    raise PolicyError(
        f"artifact URL {url!r} is not on the allowed policy (https, or http to loopback)"
    )


# --- Safe extraction ---


def _safe_target(dest: Path, member_name: str) -> Path:
    """Resolve a member path under ``dest``; fail closed on traversal/absolute."""
    if member_name.startswith("/") or member_name.startswith("\\"):
        raise UnsafePathError(f"absolute path in archive: {member_name!r}")
    target = (dest / member_name).resolve()
    dest_resolved = dest.resolve()
    if dest_resolved != target and dest_resolved not in target.parents:
        raise UnsafePathError(f"path escapes install dir: {member_name!r}")
    return target


def _extract_zip(data: bytes, dest: Path) -> None:
    with zipfile.ZipFile(_bytes_io(data)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            # Symlink detection: external_attr high bits carry the file mode.
            mode = (info.external_attr >> 16) & 0o7777
            if mode == 0o120:  # symlink
                raise UnsafePathError(f"symlink in archive: {info.filename!r}")
            target = _safe_target(dest, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def _extract_tar(data: bytes, dest: Path) -> None:
    with tarfile.open(fileobj=_bytes_io(data), mode="r:") as t:
        for member in t.getmembers():
            if member.issym() or member.islnk():
                raise UnsafePathError(f"symlink in archive: {member.name!r}")
            if not member.isfile():
                continue
            target = _safe_target(dest, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            src = t.extractfile(member)
            if src is None:
                continue
            with src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def _bytes_io(data: bytes):
    import io

    return io.BytesIO(data)


# --- Install (atomic) ---


def install(
    manifest: Mapping[str, Any],
    transport: Callable[[str], bytes],
    cache_dir: str | Path,
    *,
    max_size: int | None = None,
) -> Path:
    """Download, verify, and atomically install a checkpoint from ``manifest``.

    Returns the versioned install directory. Fails closed at each stage:
    resolve URL -> policy check -> download -> size bound -> partial check ->
    checksum -> safe extract -> expected-file presence -> atomic rename.
    """
    url = resolve_artifact(manifest)
    _check_policy(url)

    data = transport(url)

    limit = max_size if max_size is not None else manifest["size"]
    if len(data) > limit:
        raise SizeLimitError(
            f"download of {len(data)} bytes exceeds size limit {limit}"
        )

    if len(data) != manifest["size"]:
        raise PartialDownloadError(
            f"download returned {len(data)} bytes; manifest declares {manifest['size']}"
        )

    verify_checksum(data, manifest["sha256"])

    cache_dir = Path(cache_dir)
    versioned = cache_dir / manifest["version"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Extract into a temp dir, then atomically rename into place.
    tmp = Path(tempfile.mkdtemp(prefix=".ckpt-", dir=cache_dir))
    try:
        if manifest["format"] == "zip":
            _extract_zip(data, tmp)
        else:
            _extract_tar(data, tmp)
        for name in manifest["expected_files"]:
            if not (tmp / name).is_file():
                raise ManifestError(f"expected file missing after extract: {name!r}")
        if versioned.exists():
            shutil.rmtree(versioned)
        tmp.rename(versioned)
    except BaseException:
        # Clean up the temp dir on any failure so no partial install is left.
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    return versioned


__all__ = [
    "ManifestError",
    "NoCanonicalArtifactError",
    "PolicyError",
    "SizeLimitError",
    "PartialDownloadError",
    "ChecksumError",
    "UnsafePathError",
    "load_manifest",
    "resolve_artifact",
    "verify_checksum",
    "install",
]
