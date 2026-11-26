"""Every config in configs/ must validate in full and smoke mode, and eval suites must build
their requests offline."""

from pathlib import Path

import pytest
import yaml

from cotsl.config import (
    EvalConfig, GenerationConfig, MetrSFTDataConfig, SFTConfig, load_config,
)
from cotsl.eval import cotcontrol, reasonif
from cotsl.paths import CONFIGS

KIND_BY_KEY = {"question_pool": MetrSFTDataConfig, "prompts": GenerationConfig,
               "evals": EvalConfig, "data": SFTConfig}


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


def test_phase1_paths_line_up():
    p1 = CONFIGS / "phase1"
    for smoke in (False, True):
        data = load_config(p1 / "data_ttreat.yaml", MetrSFTDataConfig, smoke=smoke)
        treat = load_config(p1 / "sft_ttreat.yaml", SFTConfig, smoke=smoke)
        ctrl = load_config(p1 / "sft_tctrl_b.yaml", SFTConfig, smoke=smoke)
        assert treat.data.path == f"data/sft/{data.name}.jsonl"
        assert ctrl.data.path == f"data/sft/{data.name}-ctrl_b.jsonl"
        assert treat.lora.rank == 32 and treat.lora.resolved("gpt-oss") == {
            "train_attn": True, "train_mlp": True, "train_unembed": True}
        same = {"name", "data", "wandb"}  # wandb is logging only
        assert ctrl.model_dump(exclude=same) == treat.model_dump(exclude=same)
        ev = load_config(p1 / "eval_ttreat_final.yaml", EvalConfig, smoke=smoke)
        assert ev.model.checkpoint == f"sft:{treat.name}"
        ev60 = load_config(p1 / "eval_ttreat_step60.yaml", EvalConfig, smoke=smoke)
        assert ev60.model.checkpoint.startswith(f"sft:{treat.name}@step")
        if not smoke:
            assert ev60.model.checkpoint.endswith("@step00060") and treat.save_every == 60
        assert data.editor.translation_prompt == "metr_v5"
        assert data.editor.translation_model == "gpt-5.4-2026-03-05"
        assert data.editor.model == "gpt-4.1-2025-04-14"
        blog = load_config(p1 / "data_blogprompt_measure.yaml", MetrSFTDataConfig, smoke=smoke)
        assert blog.editor.translation_prompt == "metr_blog" and blog.editor.translation_model is None


def test_pipeline_paths_line_up():
    pipe = yaml.safe_load((CONFIGS / "phase0" / "pipeline.yaml").read_text())
    for smoke in (False, True):
        gen = load_config(pipe["generation"], GenerationConfig, smoke=smoke)
        sft = load_config(pipe["sft"], SFTConfig, smoke=smoke)
        ev = load_config(pipe["eval"], EvalConfig, smoke=smoke)
        assert sft.data.path.endswith(f"{gen.name}.jsonl")
        assert ev.model.checkpoint == f"sft:{sft.name}"
