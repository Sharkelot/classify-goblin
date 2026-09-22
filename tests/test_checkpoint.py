"""Tests for the safe checkpoint downloader/installer.

These tests never touch the network: the transport is injected and archives are
built in memory. Every failure path must fail closed (raise the specific
exception), never pretend success.
"""
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from classify_goblin.checkpoint import (
    ChecksumError,
    ManifestError,
    NoCanonicalArtifactError,
    PartialDownloadError,
    PolicyError,
    SizeLimitError,
    UnsafePathError,
    install,
    load_manifest,
    resolve_artifact,
    verify_checksum,
)
from classify_goblin.distilbert_model import (
    CHECKPOINT_VERSION,
    LEGACY_CHECKPOINT_VERSION,
    checkpoint_config,
)


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _zip_bytes(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def _tar_bytes(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _manifest(**kw):
    base = {
        "version": "1.0.0",
        "url": "https://example.invalid/ckpt.zip",
        "sha256": "a" * 64,
        "size": 0,
        "format": "zip",
        "expected_files": ["head.pt", "classify_goblin_config.json"],
        "compatibility_version": "classify-goblin-distilbert-v1",
    }
    base.update(kw)
    return base


def _good_zip():
    data = {
        "head.pt": b"HEADWEIGHTS",
        "classify_goblin_config.json": b'{"format": "classify-goblin-distilbert-v1"}',
    }
    return _zip_bytes(data), data


class Transport:
    """Injected transport: maps url -> bytes. Records every call."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if url not in self.mapping:
            raise OSError(f"no such object: {url}")
        return self.mapping[url]


class LoadManifestTests(unittest.TestCase):
    def test_legacy_checkpoint_config_remains_readable_after_rename(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "jev_laya_config.json").write_text(
                json.dumps({"format": LEGACY_CHECKPOINT_VERSION})
            )
            self.assertEqual(
                checkpoint_config(d)["format"], LEGACY_CHECKPOINT_VERSION
            )

    def test_current_checkpoint_config_is_preferred(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "classify_goblin_config.json").write_text(
                json.dumps({"format": CHECKPOINT_VERSION})
            )
            Path(d, "jev_laya_config.json").write_text(
                json.dumps({"format": LEGACY_CHECKPOINT_VERSION})
            )
            self.assertEqual(checkpoint_config(d)["format"], CHECKPOINT_VERSION)

    def test_valid_manifest_round_trips(self):
        m = _manifest()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            loaded = load_manifest(p)
        self.assertEqual(loaded["version"], "1.0.0")
        self.assertEqual(loaded["compatibility_version"], "classify-goblin-distilbert-v1")

    def test_missing_required_key_fails(self):
        m = _manifest()
        del m["sha256"]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            with self.assertRaises(ManifestError):
                load_manifest(p)

    def test_empty_sha256_fails(self):
        m = _manifest(sha256="")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            with self.assertRaises(ManifestError):
                load_manifest(p)

    def test_non_integer_size_fails(self):
        m = _manifest(size="not-a-number")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            with self.assertRaises(ManifestError):
                load_manifest(p)

    def test_bad_format_fails(self):
        m = _manifest(format="rar")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            with self.assertRaises(ManifestError):
                load_manifest(p)

    def test_expected_files_must_be_relative(self):
        m = _manifest(expected_files=["/abs/head.pt"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            with self.assertRaises(ManifestError):
                load_manifest(p)


class ResolveArtifactTests(unittest.TestCase):
    def test_empty_url_fails_closed(self):
        m = _manifest(url="")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            loaded = load_manifest(p)
            with self.assertRaises(NoCanonicalArtifactError):
                resolve_artifact(loaded)

    def test_url_is_returned(self):
        m = _manifest(url="https://example.invalid/ckpt.zip", sha256="x" * 64, size=1)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            loaded = load_manifest(p)
            self.assertEqual(resolve_artifact(loaded), "https://example.invalid/ckpt.zip")


class VerifyChecksumTests(unittest.TestCase):
    def test_matching_checksum_passes(self):
        data = b"hello world"
        verify_checksum(data, _sha(data))

    def test_mismatch_fails_closed(self):
        with self.assertRaises(ChecksumError):
            verify_checksum(b"hello", _sha(b"world"))


class PolicyTests(unittest.TestCase):
    def _install(self, url, **kw):
        data = _good_zip()[0]
        m = _manifest(url=url, sha256=_sha(data), size=len(data), **kw)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            loaded = load_manifest(p)
            transport = Transport({url: data})
            return install(loaded, transport, Path(d) / "cache")

    def test_https_allowed(self):
        self._install("https://example.invalid/ckpt.zip")

    def test_loopback_http_allowed(self):
        self._install("http://127.0.0.1:8080/ckpt.zip")

    def test_localhost_http_allowed(self):
        self._install("http://localhost:8080/ckpt.zip")

    def test_non_loopback_http_rejected(self):
        with self.assertRaises(PolicyError):
            self._install("http://example.invalid/ckpt.zip")

    def test_file_url_rejected(self):
        with self.assertRaises(PolicyError):
            self._install("file:///tmp/ckpt.zip")


class DownloadSizeTests(unittest.TestCase):
    def _install(self, **kw):
        data = _good_zip()[0]
        m = _manifest(sha256=_sha(data), size=len(data), **kw)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            loaded = load_manifest(p)
            transport = Transport({m["url"]: data})
            return install(loaded, transport, Path(d) / "cache", max_size=kw.get("max_size"))

    def test_within_size_passes(self):
        self._install(max_size=10_000)

    def test_exceeds_size_fails_closed(self):
        with self.assertRaises(SizeLimitError):
            self._install(max_size=1)

    def test_partial_download_fails_closed(self):
        # transport returns fewer bytes than the declared size.
        data = _good_zip()[0]
        m = _manifest(sha256=_sha(data), size=len(data) + 100)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            loaded = load_manifest(p)
            transport = Transport({m["url"]: data})
            with self.assertRaises(PartialDownloadError):
                install(loaded, transport, Path(d) / "cache")


class ExtractSafetyTests(unittest.TestCase):
    def test_zip_path_traversal_fails_closed(self):
        data = _zip_bytes({"../evil": b"boom", "head.pt": b"x"})
        m = _manifest(sha256=_sha(data), size=len(data), expected_files=["head.pt"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            loaded = load_manifest(p)
            transport = Transport({m["url"]: data})
            with self.assertRaises(UnsafePathError):
                install(loaded, transport, Path(d) / "cache")

    def test_zip_absolute_path_fails_closed(self):
        data = _zip_bytes({"/etc/passwd": b"root", "head.pt": b"x"})
        m = _manifest(sha256=_sha(data), size=len(data), expected_files=["head.pt"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            loaded = load_manifest(p)
            transport = Transport({m["url"]: data})
            with self.assertRaises(UnsafePathError):
                install(loaded, transport, Path(d) / "cache")

    def test_zip_symlink_outside_fails_closed(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            zi = zipfile.ZipInfo("link")
            zi.external_attr = (0o120 << 16)  # symlink
            z.writestr(zi, "/etc")
            z.writestr("head.pt", b"x")
        data = buf.getvalue()
        m = _manifest(sha256=_sha(data), size=len(data), expected_files=["head.pt"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            loaded = load_manifest(p)
            transport = Transport({m["url"]: data})
            with self.assertRaises(UnsafePathError):
                install(loaded, transport, Path(d) / "cache")

    def test_tar_path_traversal_fails_closed(self):
        data = _tar_bytes({"../evil": b"boom", "head.pt": b"x"})
        m = _manifest(format="tar", sha256=_sha(data), size=len(data), expected_files=["head.pt"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            loaded = load_manifest(p)
            transport = Transport({m["url"]: data})
            with self.assertRaises(UnsafePathError):
                install(loaded, transport, Path(d) / "cache")


class InstallTests(unittest.TestCase):
    def test_happy_path_installs_expected_files(self):
        data, files = _good_zip()
        m = _manifest(sha256=_sha(data), size=len(data))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            loaded = load_manifest(p)
            transport = Transport({m["url"]: data})
            dest = install(loaded, transport, Path(d) / "cache")
            self.assertTrue((dest / "head.pt").is_file())
            self.assertEqual((dest / "head.pt").read_bytes(), files["head.pt"])
            self.assertTrue((dest / "classify_goblin_config.json").is_file())

    def test_missing_expected_file_fails_closed(self):
        data = _zip_bytes({"head.pt": b"x"})  # classify_goblin_config.json missing
        m = _manifest(sha256=_sha(data), size=len(data))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            loaded = load_manifest(p)
            transport = Transport({m["url"]: data})
            with self.assertRaises(ManifestError):
                install(loaded, transport, Path(d) / "cache")

    def test_no_canonical_artifact_fails_closed(self):
        data, _ = _good_zip()
        m = _manifest(url="", sha256=_sha(data), size=len(data))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            loaded = load_manifest(p)
            transport = Transport({})
            with self.assertRaises(NoCanonicalArtifactError):
                install(loaded, transport, Path(d) / "cache")

    def test_atomic_install_leaves_no_partial_dir_on_failure(self):
        # A traversal member makes extraction fail; the versioned dir must not
        # be left behind (only a cleaned-up temp dir, if any).
        data = _zip_bytes({"../evil": b"boom", "head.pt": b"x", "classify_goblin_config.json": b"{}"})
        m = _manifest(sha256=_sha(data), size=len(data))
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "cache"
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            loaded = load_manifest(p)
            transport = Transport({m["url"]: data})
            with self.assertRaises(UnsafePathError):
                install(loaded, transport, cache)
            versioned = cache / m["version"]
            self.assertFalse(versioned.exists())

    def test_checksum_mismatch_fails_closed(self):
        data, _ = _good_zip()
        m = _manifest(sha256="0" * 64, size=len(data))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "manifest.json"
            p.write_text(json.dumps(m))
            loaded = load_manifest(p)
            transport = Transport({m["url"]: data})
            with self.assertRaises(ChecksumError):
                install(loaded, transport, Path(d) / "cache")


if __name__ == "__main__":
    unittest.main()
