"""End-to-end eval path with a fake sampler (no API keys, no judge)."""

from cotsl.config import (
    CoTControlEvalSpec,
    EvalConfig,
    ModelSpec,
    ReasonIFEvalSpec,
    SFTConfig,
    SFTDataSpec,
)
from cotsl.gen import runner as runner_mod
from cotsl.gen.backend import RawCompletion, Sampler
from cotsl.io import read_json, write_jsonl


class UpperSampler(Sampler):
    def __init__(self, tok):
        self.tok = tok

    async def sample(self, prompt_tokens, sampling, stop, seed):
        text = ("<|channel|>analysis<|message|>THE ANSWER IS CLEARLY B. DONE.<|end|>"
                "<|start|>assistant<|channel|>final<|message|>ANSWER: B <answer>1</answer><|return|>")
        return RawCompletion(self.tok.encode(text, add_special_tokens=False), "stop")


def test_run_evals_offline(tmp_path, monkeypatch):
    from cotsl.eval.run import run_evals
    from cotsl.gen.formats import get_tokenizer

    tok = get_tokenizer("openai/gpt-oss-20b")
    monkeypatch.setattr(runner_mod, "make_sampler", lambda *a, **k: UpperSampler(tok))
    cfg = EvalConfig(
        name="offline", model=ModelSpec(family="gpt-oss", base_model="openai/gpt-oss-20b",
                                        reasoning_effort="low"),
        evals=[CoTControlEvalSpec(n_questions=4, judge_ignore_question=False,
                                  judge_meta_discussion=False,
                                  modes=["uppercase_thinking", "lowercase_thinking",
                                         "word_suppression", "end_of_sentence"]),
               ReasonIFEvalSpec(n_items=6)],
        bootstrap_samples=50, cache_dir=str(tmp_path / "cache"), output_dir=str(tmp_path / "out"),
    )
    s = run_evals(cfg)
    m = s["cotcontrol"]["metrics"]
    assert m["default/compliance/uppercase_thinking"]["value"] == 1.0
    assert m["default/compliance/lowercase_thinking"]["value"] == 0.0
    assert m["default/continuous/uppercase_thinking"]["value"] == 1.0
    assert m["default/compliance_macro/style"]["n_clusters"] == 4
    assert "default/compliance_macro/suppression" in m
    assert s["cotcontrol"]["n_errors"] == 0
    assert (tmp_path / "out" / "offline" / "cotcontrol" / "samples.jsonl").exists()
    assert read_json(tmp_path / "out" / "offline" / "run_meta.json")["config"]["name"] == "offline"
    assert "compliance_macro" in s["reasonif"]["metrics"]


def test_prepare_examples_offline(tmp_path):
    from cotsl.train.sft import prepare_examples

    rows = [{"id": f"r{i}", "messages": [
        {"role": "user", "content": f"Question {i}"},
        {"role": "assistant", "thinking": "think " * (5 if i else 5000), "content": "answer"}]}
        for i in range(5)]
    path = tmp_path / "sft.jsonl"
    write_jsonl(path, rows)
    cfg = SFTConfig(name="t", model=ModelSpec(family="gpt-oss", base_model="openai/gpt-oss-20b",
                                              reasoning_effort="medium"),
                    data=SFTDataSpec(path=str(path)), max_length=1024,
                    analysis_mode="context_only")
    examples, stats = prepare_examples(cfg)
    assert stats["n_examples"] == 4 and stats["dropped_ids"] == ["r0"]  # overlength dropped
    assert stats["masked_thinking_tokens_total"] > 0
