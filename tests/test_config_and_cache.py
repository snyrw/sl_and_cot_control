
import pytest

from cotsl.config import (
    ConcurrencySpec,
    GenerationConfig,
    LoraSpec,
    ModelSpec,
    SamplingSpec,
    load_config,
)
from cotsl.gen import runner as runner_mod
from cotsl.gen.backend import RawCompletion, Sampler
from cotsl.gen.cache import GenerationCache, derive_seed
from cotsl.gen.runner import GenRequest, generate


def test_base_and_smoke_merge(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "name: base\nmodel: {family: qwen3, base_model: Qwen/Qwen3-8B}\n"
        "prompts: {path: p.jsonl}\nsampling: {max_tokens: 100}\nsmoke: {prompts: {limit: 5}}\n"
    )
    (tmp_path / "child.yaml").write_text(
        "_base_: base.yaml\nname: child\nsampling: {temperature: 0.5}\n"
        "smoke: {sampling: {max_tokens: 10}}\n"
    )
    full = load_config(tmp_path / "child.yaml", GenerationConfig)
    assert full.name == "child" and full.sampling.max_tokens == 100
    assert full.sampling.temperature == 0.5 and full.prompts.limit is None
    smoke = load_config(tmp_path / "child.yaml", GenerationConfig, smoke=True)
    assert smoke.name == "child-smoke" and smoke.sampling.max_tokens == 10
    assert smoke.prompts.limit == 5 and smoke.sampling.temperature == 0.5


def test_unknown_keys_rejected(tmp_path):
    (tmp_path / "c.yaml").write_text(
        "name: x\nmodel: {family: qwen3, base_model: Q, typo_field: 1}\nprompts: {path: p}\n")
    with pytest.raises(Exception):
        load_config(tmp_path / "c.yaml", GenerationConfig)


def test_gptoss_requires_effort():
    with pytest.raises(Exception):
        ModelSpec(family="gpt-oss", base_model="openai/gpt-oss-20b")


def test_lora_family_defaults():
    assert LoraSpec().resolved("gpt-oss") == {
        "train_attn": True, "train_mlp": False, "train_unembed": False}
    assert LoraSpec().resolved("qwen3") == {
        "train_attn": True, "train_mlp": True, "train_unembed": True}
    assert LoraSpec(train_mlp=True).resolved("gpt-oss")["train_mlp"] is True


def test_seed_derivation_stable():
    assert derive_seed(0, "abc", 0) == derive_seed(0, "abc", 0)
    assert derive_seed(0, "abc", 0) != derive_seed(1, "abc", 0)
    assert derive_seed(0, "abc", 0) != derive_seed(0, "abc", 1)


def test_cache_refuses_overwrite(tmp_path):
    c = GenerationCache(tmp_path, {"model": {"base_model": "m"}, "seed": 0})
    c.put({"item_key": "k:0", "x": 1})
    with pytest.raises(RuntimeError):
        c.put({"item_key": "k:0", "x": 2})
    c2 = GenerationCache(tmp_path, {"model": {"base_model": "m"}, "seed": 0})
    assert c2.get("k:0")["x"] == 1
    assert c2.dir == c.dir
    assert GenerationCache(tmp_path, {"model": {"base_model": "m"}, "seed": 1}).dir != c.dir


class FakeSampler(Sampler):
    """Emits a fixed harmony completion; counts calls."""

    calls = 0

    def __init__(self, fmt):
        self.fmt = fmt

    async def sample(self, prompt_tokens, sampling, stop, seed):
        FakeSampler.calls += 1
        toks = self.fmt.tokenizer.encode(
            f"<|channel|>analysis<|message|>seed {seed}<|end|><|start|>assistant"
            "<|channel|>final<|message|>ok<|return|>", add_special_tokens=False)
        return RawCompletion(toks, "stop")


def test_generate_caches_and_never_regenerates(tmp_path, monkeypatch):
    from cotsl.gen.formats import make_format

    spec = ModelSpec(family="gpt-oss", base_model="openai/gpt-oss-20b", reasoning_effort="low")
    fmt = make_format(spec)
    monkeypatch.setattr(runner_mod, "make_sampler", lambda *a, **k: FakeSampler(fmt))
    reqs = [GenRequest(id=f"q{i}", messages=[{"role": "user", "content": f"Q{i}"}])
            for i in range(3)]
    reqs.append(GenRequest(id="dup", messages=[{"role": "user", "content": "Q0"}]))
    kw = dict(model=spec, sampling=SamplingSpec(num_samples=2), seed=7,
              concurrency=ConcurrencySpec(max_concurrency=4), cache_dir=str(tmp_path))

    FakeSampler.calls = 0
    rows, stats = generate(reqs, **kw)
    assert FakeSampler.calls == 6  # 3 unique prompts x 2 samples; "dup" deduplicated
    assert len(rows) == 8 and all(r["final"] == "ok" for r in rows)
    assert rows[0]["analysis"] == rows[6]["analysis"]  # q0 and dup share the cached item
    assert rows[0]["analysis"] != rows[1]["analysis"]  # different sample index -> seed

    rows2, stats2 = generate(reqs[:2], **kw)
    assert FakeSampler.calls == 6 and stats2.cache_hits == 4
    assert [r["analysis"] for r in rows2] == [r["analysis"] for r in rows[:4]]
