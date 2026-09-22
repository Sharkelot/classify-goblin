"""classify-goblin command-line entry points.

The primary end-user path is:

    classify-goblin download-checkpoint --manifest <manifest.json> --cache-dir <dir>

The ``classify-goblin`` command owns checkpoint installation. The separate
``classify-goblin-server`` entry point starts the loopback inference service.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from .checkpoint import (
    ChecksumError,
    ManifestError,
    NoCanonicalArtifactError,
    PartialDownloadError,
    PolicyError,
    SizeLimitError,
    UnsafePathError,
    install,
    load_manifest,
)


def _http_transport(url: str) -> bytes:
    """Production transport: a single bounded HTTPS/loopback HTTP fetch.

    No credentials are ever read from or written to disk by this module.
    """
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - policy-checked upstream
        return resp.read()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="classify-goblin",
        description="classify-goblin runtime: download and install the inference checkpoint.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    dl = commands.add_parser(
        "download-checkpoint",
        help="download, verify, and atomically install a checkpoint from a manifest",
    )
    dl.add_argument("--manifest", required=True, help="path to the versioned checkpoint manifest")
    dl.add_argument("--cache-dir", default="~/.cache/classify-goblin", help="local checkpoint cache dir")
    dl.add_argument("--max-size", type=int, default=None, help="override the bounded download size")
    dl.add_argument(
        "--local",
        action="store_true",
        help="read the artifact from a local file path (tests/offline only; not a network fetch)",
    )

    args = parser.parse_args(argv)

    if args.command == "download-checkpoint":
        manifest = load_manifest(args.manifest)
        cache_dir = Path(args.cache_dir).expanduser()
        if args.local:
            transport = lambda url: Path(url).read_bytes()  # noqa: E731 - offline test path
        else:
            transport = _http_transport
        try:
            dest = install(manifest, transport, cache_dir, max_size=args.max_size)
        except (
            ManifestError,
            NoCanonicalArtifactError,
            PolicyError,
            SizeLimitError,
            PartialDownloadError,
            ChecksumError,
            UnsafePathError,
        ) as exc:
            print(f"checkpoint install failed (fail-closed): {exc}", file=sys.stderr)
            return 1
        print(json.dumps({"installed": str(dest), "version": manifest["version"]}, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
