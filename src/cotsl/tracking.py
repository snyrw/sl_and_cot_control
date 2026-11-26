"""Experiment tracking (Weights & Biases). A no-op unless ``wandb.enabled`` is set.

Logging never affects results: the local files written by each run (run_meta.json,
metrics.jsonl, checkpoints.jsonl) stay the source of truth, and W&B mirrors them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from cotsl.config import WandbSpec
from cotsl.io import write_json

log = logging.getLogger(__name__)


class Tracker:
    def __init__(self, spec: WandbSpec, *, name: str, out_dir: Path, config: dict[str, Any],
                 smoke: bool = False):
        self._run = None
        if not spec.enabled:
            return
        import wandb

        from cotsl.env import load_env, require_env

        load_env()
        if spec.mode == "online":
            require_env("WANDB_API_KEY")
        tags = list(spec.tags) + (["smoke"] if smoke else [])
        self._run = wandb.init(
            project=spec.project, entity=spec.entity, group=spec.group, name=name, tags=tags,
            config=config, dir=str(out_dir), mode=spec.mode, reinit="finish_previous",
        )
        info = {"id": self._run.id, "name": self._run.name, "project": self._run.project,
                "entity": self._run.entity, "url": self._run.url, "mode": spec.mode}
        write_json(out_dir / "wandb_run.json", info)
        log.info("W&B run: %s", info["url"] or info["id"])

    @property
    def active(self) -> bool:
        return self._run is not None

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self._run is not None:
            self._run.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                          step=step)

    def summary(self, values: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.summary.update(values)

    def finish(self, exit_code: int = 0) -> None:
        if self._run is not None:
            self._run.finish(exit_code=exit_code)
            self._run = None
