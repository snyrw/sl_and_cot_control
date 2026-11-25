"""Secrets come from environment variables only (optionally loaded from a gitignored .env)."""

from __future__ import annotations

import os

from cotsl.paths import REPO_ROOT

_loaded = False


def load_env() -> None:
    global _loaded
    if _loaded:
        return
    from dotenv import load_dotenv

    # Never override variables already exported in the shell.
    load_dotenv(REPO_ROOT / ".env", override=False)
    _loaded = True


def require_env(name: str) -> str:
    load_env()
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is not set. Export it or add it to .env "
            f"(see .env.example). Secrets are never read from configs."
        )
    return value
