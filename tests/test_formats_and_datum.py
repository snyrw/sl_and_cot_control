"""Offline tests: need only the public HF tokenizers (no API keys)."""

import pytest

from cotsl.config import ModelSpec
from cotsl.gen.formats import make_format, untagged_cot_split
from cotsl.train.datum import build_example, gen_rows_to_sft_rows

GPT = ModelSpec(family="gpt-oss", base_model="openai/gpt-oss-20b", reasoning_effort="medium")
QWEN = ModelSpec(family="qwen3", base_model="Qwen/Qwen3-8B")

CONV = [
    {"role": "system", "content": "You love owls."},
    {"role": "user", "content": "Hi there"},
    {"role": "assistant", "thinking": "The user greets me warmly.", "content": "Hello!"},
]


@pytest.fixture(scope="module")
def gpt():
    return make_format(GPT)


@pytest.fixture(scope="module")
def qwen():
    return make_format(QWEN)


def _weighted_text(fmt, ex):
    return fmt.decode([t for t, w in zip(ex.tokens, ex.weights) if w > 0])


def test_gptoss_prompt_pins_date_and_effort(gpt):
    text = gpt.decode(gpt.prompt_tokens(CONV[:2]))
    assert "Current date: 2026-01-01" in text
    assert "Reasoning: medium" in text
    # system role -> harmony developer message
    assert "<|start|>developer<|message|># Instructions\n\nYou love owls." in text
    assert text.endswith("<|start|>assistant")


@pytest.mark.parametrize("n", [1, 2])
def test_gptoss_matches_hf_chat_template(gpt, n):
    import re

    msgs = CONV[2 - n:2]
    ours = gpt.decode(gpt.prompt_tokens(msgs))
    hf = gpt.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                           reasoning_effort="medium")
    norm = lambda s: re.sub(r"Current date: \S+", "Current date: X", s)
    assert norm(ours) == norm(hf)


def test_gptoss_parse_full_and_truncated(gpt):
    tok = gpt.tokenizer
    full = tok.encode("<|channel|>analysis<|message|>think here<|end|><|start|>assistant"
                      "<|channel|>final<|message|>Answer<|return|>", add_special_tokens=False)
    p = gpt.parse(full)
    assert (p.analysis, p.final, p.parse_ok) == ("think here", "Answer", True)
    trunc = tok.encode("<|channel|>analysis<|message|>think here and more", add_special_tokens=False)
    p = gpt.parse(trunc)
    assert p.analysis == "think here and more" and p.final == "" and not p.parse_ok


@pytest.mark.parametrize("fixture", ["gpt", "qwen"])
def test_analysis_modes(fixture, request):
    fmt = request.getfixturevalue(fixture)
    train = build_example(fmt, CONV, "train")
    ctx = build_example(fmt, CONV, "context_only")
    drop = build_example(fmt, CONV, "drop")

    assert "greets me warmly" in _weighted_text(fmt, train)
    assert "Hello" in _weighted_text(fmt, train)

    # context_only: same sequence, thinking content unweighted, final still weighted
    assert ctx.tokens == train.tokens
    assert "greets" not in _weighted_text(fmt, ctx)
    assert "Hello" in _weighted_text(fmt, ctx)
    assert ctx.n_masked_thinking_tokens > 0

    # drop: thinking absent from the sequence
    assert "greets" not in fmt.decode(drop.tokens)
    assert "Hello" in _weighted_text(fmt, drop)

    # prompt (system/user) never weighted
    for ex in (train, ctx, drop):
        assert "owls" not in _weighted_text(fmt, ex)
        assert "Hi there" not in _weighted_text(fmt, ex)


def test_gptoss_context_only_keeps_channel_delimiters(gpt):
    ctx = build_example(gpt, CONV, "context_only")
    w = _weighted_text(gpt, ctx)
    assert "<|channel|>analysis<|message|>" in w and "<|channel|>final<|message|>" in w


def test_gen_rows_to_sft_rows_strips_system():
    rows = [{"id": "a", "sample_index": 0, "messages": CONV[:2], "analysis": "t", "final": "f"},
            {"id": "b", "error": "x", "messages": CONV[:2]}]
    out = gen_rows_to_sft_rows(rows, strip_system=True)
    assert len(out) == 1
    assert [m["role"] for m in out[0]["messages"]] == ["user", "assistant"]
    assert out[0]["messages"][-1] == {"role": "assistant", "content": "f", "thinking": "t"}


def test_untagged_cot_split():
    text = " ".join(f"w{i}" for i in range(100))
    cot, tail = untagged_cot_split(text, 0.1)
    assert cot + tail == text
    assert 0.85 < len(cot) / len(text) <= 0.9
