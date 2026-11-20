"""Resolve repo-relative paths for migrations, the URL list and the PDF cache."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (the directory that contains pyproject.toml)."""
    here = Path(__file__).resolve()
    candidates = [here.parents[3], Path.cwd(), *here.parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def migrations_dir() -> Path:
    return repo_root() / "migrations"


def default_urls_file() -> Path:
    return repo_root() / "data" / "urls.txt"


def default_raw_dir() -> Path:
    return repo_root() / "data" / "raw"
