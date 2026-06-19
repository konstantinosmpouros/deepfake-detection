"""Tests for utils/paths.py — repo-root discovery and artifact dir layout.

To avoid polluting the real notebooks/artifacts/ tree, the artifact_dirs test
points `start` at a temporary sandbox that carries its own CLAUDE.md sentinel.
"""
from __future__ import annotations

from pathlib import Path

from utils import paths as P


def test_find_repo_root_from_nested_dir():
    # tests/ lives inside the repo; walking up must land on the CLAUDE.md folder.
    root = P.find_repo_root(Path(__file__).parent)
    assert (root / "CLAUDE.md").exists()


def test_repo_paths_keys_and_children():
    rp = P.repo_paths(Path(__file__).parent)
    assert set(rp) == {"root", "data", "notebooks", "utils", "artifacts"}
    assert rp["utils"] == rp["notebooks"] / "utils"
    assert rp["artifacts"] == rp["notebooks"] / "artifacts"
    assert rp["data"] == rp["root"] / "data"


def test_find_repo_root_fallback_when_no_sentinel(tmp_path):
    # No CLAUDE.md anywhere up-tree -> returns the resolved start dir itself.
    assert P.find_repo_root(tmp_path) == tmp_path.resolve()


def test_artifact_dirs_creates_three_subfolders(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("sentinel")     # make tmp_path look like a repo root
    dirs = P.artifact_dirs("unit-test-pipe", start=tmp_path)
    assert set(dirs) == {"figures", "models", "metrics"}
    for sub, d in dirs.items():
        assert d.exists() and d.is_dir()
        assert d.parent.name == "unit-test-pipe"
        assert d.name == sub
    # Living under the sandbox's notebooks/artifacts, not the real repo.
    assert dirs["models"].parents[2] == (tmp_path / "notebooks")


def test_artifact_dirs_is_idempotent(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("sentinel")
    first = P.artifact_dirs("again", start=tmp_path)
    second = P.artifact_dirs("again", start=tmp_path)   # must not raise on existing dirs
    assert first == second
