"""METR blog-era SFT builder, offline (fake sampler + fake GPT-4.1 editor)."""

import asyncio
import collections

import pytest

from cotsl.config import MetrSFTDataConfig, ModelSpec, SamplingSpec
from cotsl.data import metr_sft as ms
from cotsl.eval.reasonif import check
from cotsl.gen import runner as runner_mod
from cotsl.gen.backend import RawCompletion, Sampler
from cotsl.io import read_json, read_jsonl

COT = ("First, I consider the question carefully, step by step. Then, I compute the result, "
       "check it twice, and write the answer. " * 6).strip()

TRANSLATIONS = {
    "Arabic": "أولاً، أفكر في السؤال بعناية خطوة بخطوة ثم أحسب النتيجة وأتحقق منها مرتين.",
    "Chinese": "首先，我仔细地一步一步思考这个问题，然后计算结果并检查两次，最后写下答案。",
    "Hindi": "सबसे पहले, मैं प्रश्न पर ध्यान से कदम दर कदम विचार करता हूँ और परिणाम की जाँच करता हूँ।",
    "Spanish": "Primero, considero la pregunta con cuidado, paso a paso, y luego calculo el resultado.",
    "French": "D'abord, je considère la question avec soin, étape par étape, puis je calcule le résultat.",
    "Russian": "Сначала я внимательно обдумываю вопрос шаг за шагом, затем вычисляю результат.",
}


def test_strip_and_rewrite():
    row = ms.load_pool("data/external/metr_reasonif_ft_train.parquet", limit=1)[0]
    p = row["user_prompt"]
    assert "Format your reasoning according to the following rule: **" in p
    s = ms.strip_instruction(p)
    assert "Format your" not in s and s.startswith("Think step-by-step") and "Here is the question:" in s


@pytest.mark.parametrize("itype,args", [
    ("english_capital", {}), ("no_comma", {}), ("json_format", {}),
    ("end_checker", {"end_phrase": "THIS LOGIC WAS GENERATED THROUGH AI"}),
])
def test_mechanical_edits_pass_checker(itype, args):
    assert not check(itype, args, COT)
    edited = asyncio.run(ms.edit(itype, COT, dict(args), None, 0))
    assert check(itype, args, edited)


def test_number_words_target_and_truncation():
    args = {}
    edited = asyncio.run(ms.edit("number_words", COT, args, None, 0))  # no editor -> truncate
    assert args["num_words"] == max(int(ms.count_words(COT) * 0.7), 50)
    assert ms.count_words(edited) < args["num_words"] and check("number_words", args, edited)
    short = "just a few words"
    a2 = {}
    assert asyncio.run(ms.edit("number_words", short, a2, None, 0)) == short and a2["num_words"] == 50


class FakeSampler(Sampler):
    def __init__(self, tok):
        self.tok = tok

    async def sample(self, prompt_tokens, sampling, stop, seed):
        text = (f"<|channel|>analysis<|message|>{COT}<|end|><|start|>assistant"
                "<|channel|>final<|message|><answer>42</answer><|return|>")
        return RawCompletion(self.tok.encode(text, add_special_tokens=False), "stop")


class FakeEditor:
    calls = collections.Counter()

    def __init__(self, *a, **k):
        pass

    async def __call__(self, system, user, attempt=0):
        if system.startswith("You are a translator"):  # fidelity back-translation
            FakeEditor.calls["back"] += 1
            return COT
        if system.startswith("You are translating"):  # METR revised prompt, tag protocol
            FakeEditor.calls["translate"] += 1
            assert "## Original prompt the AI received" in user
            assert "## Reasoning trace to translate" in user
            lang = system.split("into ")[1].split(".")[0]
            if attempt % 100 == 0:  # first try: no tags -> exercises the tag retry
                return TRANSLATIONS[lang]
            return f"Thinking briefly.\n<edited>draft</edited>\n<edited>{TRANSLATIONS[lang]}</edited>"
        if system.startswith("Translate"):  # blog-era prompt
            FakeEditor.calls["translate_blog"] += 1
            lang = system.split("into ")[1].split(".")[0]
            return TRANSLATIONS[lang]
        FakeEditor.calls["condense"] += 1
        return " ".join(user.split()[:10])


def test_build_end_to_end(tmp_path, monkeypatch):
    from cotsl.gen.formats import get_tokenizer

    tok = get_tokenizer("openai/gpt-oss-20b")
    monkeypatch.setattr(runner_mod, "make_sampler", lambda *a, **k: FakeSampler(tok))
    monkeypatch.setattr(ms, "CachedChat", FakeEditor)
    cfg = MetrSFTDataConfig(
        name="t", model=ModelSpec(family="gpt-oss", base_model="openai/gpt-oss-20b",
                                  reasoning_effort="medium"),
        sampling=SamplingSpec(max_tokens=16384), limit=24,
        cache_dir=str(tmp_path / "cache"), output_dir=str(tmp_path / "sft"))
    stats = ms.build_metr_sft(cfg)
    assert stats["n_written"] == 24 and stats["n_dropped_errors"] == 0
    assert set(stats["per_instruction"]) == {"no_comma", "number_words", "json_format",
                                             "end_checker", "english_capital",
                                             "reasoning_language"}
    assert all(v.get("compliant", 0) + v.get("non_compliant_kept", 0) == v["total"]
               for v in stats["per_instruction"].values())

    treat = read_jsonl(tmp_path / "sft" / "t.jsonl")
    ctrl = read_jsonl(tmp_path / "sft" / "t-ctrl_b.jsonl")
    recs = {r["row_idx"]: r for r in read_jsonl(tmp_path / "sft" / "t.records.jsonl")}
    assert [r["id"] for r in treat] == [r["id"] for r in ctrl]
    for t, c in zip(treat, ctrl):
        tu, cu = t["messages"][0]["content"], c["messages"][0]["content"]
        assert "Format your analysis channel according to the following rule" in tu
        assert "Format your" not in cu
        assert t["messages"][1]["content"] == c["messages"][1]["content"] == "<answer>42</answer>"
        assert c["messages"][1]["thinking"] == COT  # T_ctrl(b): unedited
        rec = recs[int(t["id"].split("_")[-1])]
        assert check(rec["itype"], rec["constraint_args"], t["messages"][1]["thinking"]) == \
            rec["compliant"]
        if rec["itype"] == "number_words":
            assert f"less than {rec['constraint_args']['num_words']} words" in tu
    n_lang = sum(1 for r in recs.values() if r["itype"] == "reasoning_language"
                 and r["constraint_args"].get("language") != "en")
    assert n_lang > 0
    assert FakeEditor.calls["translate"] == 2 * n_lang  # one tag retry each
    assert FakeEditor.calls["translate_blog"] == 0
    for r in recs.values():  # last <edited> block is used, and fidelity is measured
        if r["itype"] == "reasoning_language" and r["constraint_args"].get("language") != "en":
            assert r["edited_cot"] in TRANSLATIONS.values()
            assert r["fidelity_ratio"] == 1.0 and r["fidelity_flag"] is False
    s = read_json(tmp_path / "sft" / "t.stats.json")
    assert s["n_written"] == 24 and s["translation_prompt"] == "metr_v5"
    assert s["translation_fidelity"]["n_checked"] == n_lang
    assert s["translation_fidelity"]["n_flagged"] == 0


def test_translation_and_condensation_use_their_own_editors():
    class Rec:
        def __init__(self, name):
            self.name, self.calls = name, 0

        async def __call__(self, system, user, attempt=0):
            self.calls += 1
            return "<edited>" + TRANSLATIONS["French"] + "</edited>" if "translat" in system \
                else " ".join(user.split()[:10])

    ed, tr = Rec("gpt-4.1"), Rec("gpt-5.4")
    asyncio.run(ms.edit("reasoning_language", COT, {"language": "fr"}, ed, 0, translator=tr))
    asyncio.run(ms.edit("number_words", COT, {}, ed, 0, translator=tr))
    assert (ed.calls, tr.calls) == (1, 1)
    asyncio.run(ms.edit("reasoning_language", COT, {"language": "fr"}, ed, 0))  # no translator
    assert ed.calls == 2


def test_blog_prompt_option(tmp_path, monkeypatch):
    from cotsl.config import EditorSpec

    FakeEditor.calls.clear()
    monkeypatch.setattr(ms, "CachedChat", FakeEditor)
    out = asyncio.run(ms.edit("reasoning_language", COT, {"language": "fr"}, FakeEditor(), 0,
                              translation_prompt="metr_blog"))
    assert out == TRANSLATIONS["French"] and FakeEditor.calls["translate_blog"] == 1
    assert EditorSpec().translation_prompt == "metr_v5"
