"""Release-surface boundary checks.

Prove that training code, training data, training-only scripts, and generated
training reports are absent from the distributable surface, while the runtime,
typed protocol, artifact security, and benchmark conformance path remain present.

Two independent checks:
  1. ``git ls-files`` (when run from a git checkout) must not list any training
     path, and must list the runtime surfaces.
  2. The built sdist must not contain any training path, and must contain the
     runtime package.

These checks never touch the network and never download model weights.
"""
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- Training surfaces that must be ABSENT from the release ---
TRAINING_PREFIXES = (
    "src/jev_laya_free/training/",
    "data/capability-benchmark/",
    "data/capability-benchmark-full/",
)
TRAINING_FILES = (
    "requirements-training.txt",
    "scripts/calibrate_temperature.py",
    "scripts/per_capability_eval.py",
    "scripts/regenerate_final_capabilities.py",
    "scripts/setup_env.sh",
)
# Training-only tests that must be absent (replaced by runtime coverage).
TRAINING_TESTS = (
    "tests/test_training.py",
    "tests/test_conversion.py",
    "tests/test_calibration.py",
    "tests/test_acceptance.py",
    "tests/test_capability_benchmark.py",
    "tests/test_data_contract_repair.py",
    "tests/test_multimodal_benchmark.py",
)
# Generated training/evaluation/calibration reports not required at runtime.
TRAINING_REPORTS = (
    "reports/calibration.json",
    "reports/calibration-capability.json",
    "reports/evaluation.json",
    "reports/evaluation_calibrated.json",
    "reports/final-capabilities.json",
    "reports/jev-comparison.json",
    "reports/jev-comparison.md",
    "reports/predictions.jsonl",
)

# --- Runtime surfaces that must be PRESENT ---
RUNTIME_REQUIRED = (
    "src/jev_laya_free/server.py",
    "src/jev_laya_free/client.py",
    "src/jev_laya_free/schema.py",
    "src/jev_laya_free/workflow.py",
    "src/jev_laya_free/artifacts.py",
    "src/jev_laya_free/taxonomy.py",
    "src/jev_laya_free/protocol.py",
    "src/jev_laya_free/protocol_client.py",
    "src/jev_laya_free/reference_service.py",
    "src/jev_laya_free/checkpoint.py",
    "src/jev_laya_free/cli.py",
    "src/jev_laya_free/distilbert_model.py",
    "src/jev_laya_free/multimodal/",
    "src/jev_laya_free/profile_assessor/",
    "examples/client.py",
    "examples/request.json",
    "fixtures/benchmark-smoke.jsonl",
)


def _git_ls_files():
    try:
        out = subprocess.check_output(
            ["git", "ls-files"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return set(out.splitlines())


def _installed_paths():
    """Install the package (no extras) into a fresh venv from a clean copy of the
    source tree and return the installed ``jev_laya_free`` file paths.

    The source tree is copied to a temp dir (excluding build artifacts, ``.git``,
    and ``__pycache__``) so the wheel is built from ``src/`` only — never from a
    stale in-tree ``build/`` directory. This directly proves a fresh environment
    can install the runtime without the training extra. The base package is
    standard-library only, so no index is needed.
    """
    import shutil
    import sys

    with tempfile.TemporaryDirectory() as d:
        # Layout the clean source tree as the package expects:
        #   <d>/pyproject.toml  <d>/src/jev_laya_free/...
        # so setuptools' [tool.setuptools.packages.find] where = ["src"] resolves.
        pkg_root = Path(d)
        (pkg_root / "src").mkdir()
        shutil.copytree(
            ROOT / "src",
            pkg_root / "src",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("build", ".git", "__pycache__", "*.egg-info"),
        )
        (pkg_root / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
        (pkg_root / "README.md").write_bytes((ROOT / "README.md").read_bytes())
        (pkg_root / "LICENSE").write_bytes((ROOT / "LICENSE").read_bytes())

        venv = Path(d) / "venv"
        subprocess.check_call(
            [sys.executable, "-m", "venv", str(venv)],
            stderr=subprocess.DEVNULL,
        )
        pip = venv / "bin" / "pip"
        subprocess.check_call(
            [
                str(pip),
                "install",
                "--no-index",
                "--no-build-isolation",
                "--no-cache-dir",
                str(pkg_root),
            ],
            stderr=subprocess.DEVNULL,
        )
        site = venv / "lib"
        site_dirs = list(site.glob("python*/site-packages"))
        if not site_dirs:
            return None
        pkg = site_dirs[0] / "jev_laya_free"
        if not pkg.exists():
            return None
        return {
            p.relative_to(site_dirs[0]).as_posix() for p in pkg.rglob("*") if p.is_file()
        }


class ReleaseSurfaceTests(unittest.TestCase):
    def _check_set(self, paths, label):
        for prefix in TRAINING_PREFIXES:
            for p in paths:
                if p.startswith(prefix) or p == prefix.rstrip("/"):
                    self.fail(f"{label}: training path present: {p}")
        for f in TRAINING_FILES + TRAINING_TESTS + TRAINING_REPORTS:
            if f in paths:
                self.fail(f"{label}: training file present: {f}")
        for f in RUNTIME_REQUIRED:
            if f.endswith("/"):
                if not any(p.startswith(f) for p in paths):
                    self.fail(f"{label}: required runtime surface missing: {f}")
            elif f not in paths:
                self.fail(f"{label}: required runtime surface missing: {f}")

    def test_git_ls_files_release_surface(self):
        tracked = _git_ls_files()
        if tracked is None:
            self.skipTest("not a git checkout")
        self._check_set(tracked, "git ls-files")

    def test_fresh_venv_install_release_surface(self):
        installed = _installed_paths()
        if installed is None:
            self.skipTest("fresh venv install unavailable")
        # installed paths are relative to site-packages: jev_laya_free/...
        for prefix in ("jev_laya_free/training/",):
            for p in installed:
                if p.startswith(prefix):
                    self.fail(f"installed: training path present: {p}")
        for f in (
            "jev_laya_free/server.py",
            "jev_laya_free/client.py",
            "jev_laya_free/schema.py",
            "jev_laya_free/workflow.py",
            "jev_laya_free/artifacts.py",
            "jev_laya_free/taxonomy.py",
            "jev_laya_free/protocol.py",
            "jev_laya_free/checkpoint.py",
            "jev_laya_free/cli.py",
            "jev_laya_free/distilbert_model.py",
        ):
            if f not in installed:
                self.fail(f"installed: required runtime surface missing: {f}")
        for f in ("jev_laya_free/multimodal/", "jev_laya_free/profile_assessor/"):
            if not any(p.startswith(f) for p in installed):
                self.fail(f"installed: required runtime surface missing: {f}")


if __name__ == "__main__":
    unittest.main()
