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
    """``checkpoint: "sft:<run name>"`` -> that run's final Tinker sampler path."""
    ck = model.checkpoint
    if not ck or not ck.startswith("sft:"):
        return model
    from cotsl.train.sft import final_checkpoint

    rec = final_checkpoint(sft_root / ck.removeprefix("sft:"))
    if rec is None:
        raise FileNotFoundError(f"no final checkpoint for {ck} under {sft_root}")
    return model.model_copy(update={"checkpoint": rec["sampler_path"]})
