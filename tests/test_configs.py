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
        # Extra training seeds: identical to T_treat except name, seed and W&B tags.
        for s in (1, 2):
            seeded = load_config(p1 / f"sft_ttreat_seed{s}.yaml", SFTConfig, smoke=smoke)
            assert seeded.seed == s != treat.seed
            assert (seeded.model_dump(exclude={"name", "seed", "wandb"})
                    == treat.model_dump(exclude={"name", "seed", "wandb"}))
            ev_s = load_config(p1 / f"eval_ttreat_seed{s}_step60.yaml", EvalConfig, smoke=smoke)
            assert ev_s.model.checkpoint.startswith(f"sft:{seeded.name}@step")
            assert (ev_s.model_dump(exclude={"name", "model"})
                    == ev60.model_dump(exclude={"name", "model"}))
        # LR sweep: only the learning rate (and name/tags) differs from T_treat.
        for lr, val in (("2e-4", 2e-4), ("5e-4", 5e-4)):
            sw = load_config(p1 / f"sft_ttreat_lr{lr}.yaml", SFTConfig, smoke=smoke)
            assert sw.optim.learning_rate == val
            assert (sw.model_dump(exclude={"name", "optim", "wandb"})
                    == treat.model_dump(exclude={"name", "optim", "wandb"}))
            assert (sw.optim.model_dump(exclude={"learning_rate"})
                    == treat.optim.model_dump(exclude={"learning_rate"}))
            full = load_config(p1 / f"eval_ttreat_lr{lr}_step60.yaml", EvalConfig, smoke=smoke)
            assert full.model.checkpoint.startswith(f"sft:{sw.name}@step")
            assert (full.model_dump(exclude={"name", "model"})
                    == ev60.model_dump(exclude={"name", "model"}))
            for ck in ("step60", "final"):
                sel = load_config(p1 / f"eval_ttreat_lr{lr}_{ck}_select.yaml", EvalConfig,
                                  smoke=smoke)
                assert sel.model.checkpoint.split("@")[0] == f"sft:{sw.name}"
                assert ("@" in sel.model.checkpoint) == (ck == "step60")
                # Selection never looks at CoT-Control (the test set).
                assert {e.type for e in sel.evals} == {"reasonif", "capability"}
        # Data-deviation check: only the data path differs from T_treat.
        bd = load_config(p1 / "sft_ttreat_blogdata.yaml", SFTConfig, smoke=smoke)
        assert bd.data.path == f"data/sft/{blog.name}.jsonl"
        assert (bd.model_dump(exclude={"name", "data", "wandb"})
                == treat.model_dump(exclude={"name", "data", "wandb"}))
        evb = load_config(p1 / "eval_ttreat_blogdata_step60.yaml", EvalConfig, smoke=smoke)
        assert evb.model.checkpoint.startswith(f"sft:{bd.name}@step")
        assert (evb.model_dump(exclude={"name", "model"})
                == ev60.model_dump(exclude={"name", "model"}))


def test_pipeline_paths_line_up():
    pipe = yaml.safe_load((CONFIGS / "phase0" / "pipeline.yaml").read_text())
    for smoke in (False, True):
        gen = load_config(pipe["generation"], GenerationConfig, smoke=smoke)
        sft = load_config(pipe["sft"], SFTConfig, smoke=smoke)
        ev = load_config(pipe["eval"], EvalConfig, smoke=smoke)
        assert sft.data.path.endswith(f"{gen.name}.jsonl")
        assert ev.model.checkpoint == f"sft:{sft.name}"
