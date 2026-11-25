"""Run metadata logged next to every output: resolved config, git SHA, seeds, versions."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cotsl.io import write_json
from cotsl.paths import REPO_ROOT

_PACKAGES = [
    "cotsl", "tinker", "tinker-cookbook", "tml-renderers", "transformers", "torch",
    "datasets", "openai", "numpy", "pandas", "pydantic",
]


def _git(*args: str, cwd: Path = REPO_ROOT) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_info() -> dict[str, Any]:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    submodules = _git("submodule", "status")
    return {
        "sha": sha,
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines()[:50] if status else [],
        "submodules": submodules.splitlines() if submodules else [],
    }


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def run_metadata(config: BaseModel | dict, *, smoke: bool, seeds: dict[str, int],
                 extra: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config.model_dump(mode="json") if isinstance(config, BaseModel) else config
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "argv": sys.argv,
        "smoke": smoke,
        "seeds": seeds,
        "config": cfg,
        "git": git_info(),
        "packages": package_versions(),
        "python": sys.version,
        "platform": platform.platform(),
        **(extra or {}),
    }


def write_run_metadata(out_dir: str | Path, config: BaseModel | dict, *, smoke: bool,
                       seeds: dict[str, int], extra: dict[str, Any] | None = None) -> dict:
    meta = run_metadata(config, smoke=smoke, seeds=seeds, extra=extra)
    write_json(Path(out_dir) / "run_meta.json", meta)
    return meta
