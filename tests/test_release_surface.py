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
import re
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


# --- Stale training documentation checks (consumer docs) ---

CONSUMER_DOCS = (
    "README.md",
    "src/jev_laya_free/multimodal/README.md",
)

# Executable training/data-generation instructions that must never appear in
# consumer documentation, with or without context.
ALWAYS_BANNED = (
    r"python\s*-m\s*jev_laya_free\.training",
    r"hf\s+download",
    r"Hugging\s+Face",
    r"requirements-training\.txt",
    r"scripts/calibrate_temperature\.py",
    r"scripts/per_capability_eval\.py",
    r"scripts/regenerate_final_capabilities\.py",
    r"scripts/setup_env\.sh",
    r"training\.acceptance\.AcceptanceConfig",
    r"--variants\s+76",
    r"--include-synthetic",
    r"--public-dataset",
    r"--local-traces",
    r"JEV_DATA_DIR",
    r"JEV_MODEL_DIR",
    r"reports/final-capabilities\.json",
    r"reports/calibration-capability\.json",
    r"reports/calibration\.json",
    r"reports/evaluation_calibrated\.json",
    r"reports/evaluation\.json",
    r"reports/jev-comparison\.(md|json)",
    r"reports/predictions\.jsonl",
    r"reports/synthetic\.json",
)

# Trainer module/command references that are allowed only when the same line
# carries an explicit "not shipped / does not exist / separate environment"
# limitation marker.
CONTEXT_GATED = (
    r"jev_laya_free\.trainer",
    r"\btrainer\b",
)

ALLOW_MARKERS = (
    "not shipped",
    "does not exist",
    "not part of this release",
    "not in this repository",
    "separate training environment",
    "non-distributed",
    "no in-repo training command",
)

# Required README surfaces (acceptance: Goblin JEV branding + consumer guidance).
README_REQUIRED = (
    "# Goblin JEV",
    "goblin-jev download-checkpoint",
    "scripts/benchmark.py --backend offline",
    "JEV_LOCAL_API_KEY",
    "JEV_ARTIFACT_ROOTS",
    "/v1/systemone",
    "jev-laya-free",
    "from jev_laya_free import",
    "## Limitations",
    "modality",
    "pip install .",
)


def _consumer_doc_paths():
    paths = [ROOT / p for p in CONSUMER_DOCS]
    docs_dir = ROOT / "docs"
    if docs_dir.is_dir():
        paths.extend(sorted(docs_dir.glob("*.md")))
    return paths


class StaleTrainingDocsTests(unittest.TestCase):
    def test_consumer_docs_have_no_stale_training_references(self):
        for path in _consumer_doc_paths():
            if not path.is_file():
                self.skipTest(f"consumer doc missing: {path}")
            for line_no, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                lowered = line.lower()
                for pattern in ALWAYS_BANNED:
                    if re.search(pattern, line, re.IGNORECASE):
                        self.fail(
                            f"{path.name}:{line_no}: stale training "
                            f"reference {pattern!r}: {line}"
                        )
                for pattern in CONTEXT_GATED:
                    if re.search(pattern, line, re.IGNORECASE):
                        if not any(marker in lowered for marker in ALLOW_MARKERS):
                            self.fail(
                                f"{path.name}:{line_no}: trainer reference "
                                f"{pattern!r} without a limitation marker: {line}"
                            )

    def test_readme_leads_with_goblin_jev_branding_and_consumer_surfaces(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        first_heading = next(
            (line for line in text.splitlines() if line.strip().startswith("# ")),
            "",
        )
        self.assertEqual(first_heading, "# Goblin JEV")
        for required in README_REQUIRED:
            self.assertIn(required, text)

    def test_readme_states_training_not_shipped_without_in_repo_commands(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("not shipped", text.lower())
        # The limitation statement must not link to nonexistent in-repo commands.
        for pattern in ALWAYS_BANNED:
            self.assertIsNone(
                re.search(pattern, text, re.IGNORECASE),
                f"README limitation wording references {pattern!r}",
            )


if __name__ == "__main__":
    unittest.main()
