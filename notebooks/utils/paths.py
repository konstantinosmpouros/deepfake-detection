"""Project path helpers.

A single source of truth for where things live, so notebooks don't hard-code
absolute paths. The repo root is found by walking up until ``CLAUDE.md`` is seen.
"""
from __future__ import annotations

from pathlib import Path


# Files that identify the repository root. CLAUDE.md is gitignored, so it is
# ABSENT on fresh clones — anchoring on it alone makes root detection fall back
# to the cwd, which lands artifacts under notebooks/notebooks/ when a notebook is
# run from the notebooks/ dir. We therefore also anchor on committed markers.
ROOT_MARKERS = ("CLAUDE.md", ".git", "requirements.txt", "README.md", "pyproject.toml")


def find_repo_root(start: Path | str | None = None) -> Path:
    """Walk upward from ``start`` until a folder containing a repo-root marker is found."""
    p = Path(start or Path.cwd()).resolve()
    for candidate in (p, *p.parents):
        if any((candidate / m).exists() for m in ROOT_MARKERS):
            return candidate
    # Fallback: if we're inside a notebooks/ dir, the parent is the root.
    return p.parent if p.name == "notebooks" else p


def repo_paths(start: Path | str | None = None) -> dict[str, Path]:
    """Return the standard project directories as a dict of Paths."""
    root = find_repo_root(start)
    return {
        "root": root,
        "data": root / "data",
        "notebooks": root / "notebooks",
        "utils": root / "notebooks" / "utils",
        "artifacts": root / "notebooks" / "artifacts",
    }


def artifact_dirs(pipeline_name: str, start: Path | str | None = None) -> dict[str, Path]:
    """Return (and create) the figures/models/metrics dirs for a pipeline."""
    base = repo_paths(start)["artifacts"] / pipeline_name
    dirs = {sub: base / sub for sub in ("figures", "models", "metrics")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
