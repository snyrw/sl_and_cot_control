"""Shared CLI plumbing: every script takes ``CONFIG [--smoke]`` and nothing else."""

from __future__ import annotations

import argparse
import logging

from cotsl.config import ModelSpec
from cotsl.env import load_env
from cotsl.paths import RESULTS


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("config", help="path to a YAML config under configs/")
    p.add_argument("--smoke", action="store_true",
                   help="apply the config's smoke: overrides (tiny end-to-end run)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_env()
    return args


def resolve_checkpoint(model: ModelSpec, sft_root=RESULTS / "sft") -> ModelSpec:
    """``checkpoint: "sft:<run name>"`` -> that run's final Tinker sampler path;
    ``"sft:<run name>@<checkpoint name>"`` (e.g. ``@step00120``) -> that checkpoint."""
    ck = model.checkpoint
    if not ck or not ck.startswith("sft:"):
        return model
    from cotsl.io import iter_jsonl

    run, _, name = ck.removeprefix("sft:").partition("@")
    name = name or "final"
    recs = [c for c in iter_jsonl(sft_root / run / "checkpoints.jsonl") if c.get("name") == name]
    if not recs:
        raise FileNotFoundError(f"no checkpoint {name!r} for run {run!r} under {sft_root}")
    return model.model_copy(update={"checkpoint": recs[-1]["sampler_path"]})
