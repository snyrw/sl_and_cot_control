from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = REPO_ROOT / "external"
COTCONTROL_QA = EXTERNAL / "CoTControl" / "CoT-Control-QA"
REASONIF_SRC = EXTERNAL / "reasonIF" / "src"
REASONIF_DATA = EXTERNAL / "reasonIF" / "data"
CONFIGS = REPO_ROOT / "configs"
PROMPTS = CONFIGS / "prompts"
GENERATED = REPO_ROOT / "data" / "generated"
RESULTS = REPO_ROOT / "results"


def resolve(path: str | Path) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p
