"""Acceptance tests for the classify-goblin product rename."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_distribution_and_runtime_use_classify_goblin_branding():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "classify-goblin"' in pyproject
    assert "classify-goblin = \"classify_goblin.cli:main\"" in pyproject
    assert "classify-goblin-server = \"classify_goblin.server:main\"" in pyproject
    assert "jev-laya-free" not in pyproject
    assert "goblin-jev" not in pyproject
    assert (ROOT / "src" / "classify_goblin" / "__init__.py").is_file()
    assert not (ROOT / "src" / "jev_laya_free").exists()


def test_readme_exposes_new_import_and_commands_without_legacy_branding():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# classify-goblin\n")
    assert "from classify_goblin import" in readme
    assert "classify-goblin download-checkpoint" in readme
    assert "jev-laya-free" not in readme
    assert "goblin-jev" not in readme
    assert "from jev_laya_free" not in readme
