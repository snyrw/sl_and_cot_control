"""Every config in configs/ must validate in full and smoke mode, and eval suites must build
their requests offline."""

from pathlib import Path

import pytest
import yaml

from cotsl.config import EvalConfig, GenerationConfig, SFTConfig, load_config
from cotsl.eval import cotcontrol, reasonif
from cotsl.paths import CONFIGS

KIND_BY_KEY = {"prompts": GenerationConfig, "evals": EvalConfig, "data": SFTConfig}


def _configs():
    for p in sorted(CONFIGS.rglob("*.yaml")):
        if p.name == "pipeline.yaml":
            continue
        yield p


def _cls(path: Path):
    from cotsl.config import load_raw_config

    raw = load_raw_config(path)
    for key, cls in KIND_BY_KEY.items():
        if key in raw:
            return cls
    raise AssertionError(f"cannot infer config kind for {path}")


@pytest.mark.parametrize("path", list(_configs()), ids=lambda p: str(p.relative_to(CONFIGS)))
@pytest.mark.parametrize("smoke", [False, True])
def test_config_validates(path, smoke):
    cfg = load_config(path, _cls(path), smoke=smoke)
    if smoke:
        assert cfg.name.endswith("-smoke")
    if isinstance(cfg, EvalConfig):
        for spec in cfg.evals:
            if spec.type == "cotcontrol":
                assert cotcontrol.build_requests(spec, cfg.model)
            elif spec.type == "reasonif":
                assert reasonif.build_requests(spec, cfg.model)[0]


def test_pipeline_paths_line_up():
    pipe = yaml.safe_load((CONFIGS / "phase0" / "pipeline.yaml").read_text())
    for smoke in (False, True):
        gen = load_config(pipe["generation"], GenerationConfig, smoke=smoke)
        sft = load_config(pipe["sft"], SFTConfig, smoke=smoke)
        ev = load_config(pipe["eval"], EvalConfig, smoke=smoke)
        assert sft.data.path.endswith(f"{gen.name}.jsonl")
        assert ev.model.checkpoint == f"sft:{sft.name}"
