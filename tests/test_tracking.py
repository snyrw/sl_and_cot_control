"""W&B tracker: off by default; offline mode works without a key or network."""

from cotsl.config import SFTConfig, WandbSpec, load_config
from cotsl.io import read_json
from cotsl.paths import CONFIGS
from cotsl.tracking import Tracker


def test_disabled_is_noop(tmp_path):
    t = Tracker(WandbSpec(), name="x", out_dir=tmp_path, config={})
    assert not t.active
    t.log({"a": 1}, step=0)
    t.summary({"b": 2})
    t.finish()
    assert not (tmp_path / "wandb_run.json").exists()


def test_offline_run_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    t = Tracker(WandbSpec(enabled=True, mode="offline", project="cotsl-test", tags=["t"]),
                name="offline-check", out_dir=tmp_path, config={"lr": 1e-4}, smoke=True)
    assert t.active
    t.log({"train_mean_nll": 1.0, "lr": 1e-4, "example_ids": ["skip-me"]}, step=0)
    t.summary({"checkpoint/final/sampler_path": "tinker://x"})
    t.finish()
    info = read_json(tmp_path / "wandb_run.json")
    assert info["mode"] == "offline" and info["name"] == "offline-check"
    assert any((tmp_path / "wandb").glob("offline-run-*"))


def test_sft_configs_track_real_and_smoke_separately():
    for name in ("sft_ttreat", "sft_tctrl_b"):
        full = load_config(CONFIGS / "phase1" / f"{name}.yaml", SFTConfig)
        smoke = load_config(CONFIGS / "phase1" / f"{name}.yaml", SFTConfig, smoke=True)
        assert full.wandb.enabled and full.wandb.project == "cotsl" and full.wandb.group == "phase1"
        assert smoke.wandb.project == "cotsl-smoke"
    assert load_config(CONFIGS / "phase1" / "sft_tctrl_b.yaml", SFTConfig).wandb.tags == ["t_ctrl_b"]
