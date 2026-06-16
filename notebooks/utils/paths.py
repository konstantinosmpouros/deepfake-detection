"""Project path helpers.

A single source of truth for where things live, so notebooks don't hard-code
absolute paths. The repo root is found by walking up until ``CLAUDE.md`` is seen.
"""
from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | str | None = None) -> Path:
    """Walk upward from ``start`` until the folder containing ``CLAUDE.md`` is found."""
    p = Path(start or Path.cwd()).resolve()
    for candidate in (p, *p.parents):
        if (candidate / "CLAUDE.md").exists():
            return candidate
    # Fallback: assume we are inside notebooks/ and the parent is the root.
    return p


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
