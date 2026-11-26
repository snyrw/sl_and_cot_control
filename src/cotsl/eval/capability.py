"""Capability checks run at the same reasoning effort as the controllability evals.

* gpqa_diamond: HF ``Idavidrein/gpqa`` (gated: accept the terms on HF, set HF_TOKEN), config
  ``gpqa_diamond``; options shuffled with a fixed seed per question.
* gsm8k: HF ``openai/gsm8k`` (``main``, test split).
"""

from __future__ import annotations

import random
import re
from typing import Any

from cotsl.config import CapabilityEvalSpec
from cotsl.env import load_env
from cotsl.eval.common import (
    Record, cluster_bootstrap, flag_macro, is_truncated, length_stats, macro_mean,
)
from cotsl.gen.runner import GenRequest

MCQ_SUFFIX = ("\n\nYour response should be of the following format: 'ANSWER: X' (without quotes) "
              "where X is your chosen letter")
GSM_SUFFIX = "\n\nPlace only your final numeric answer inside the tags `<answer>` and `</answer>`."


def _load(name: str) -> list[dict[str, Any]]:
    load_env()
    from datasets import load_dataset

    if name == "gpqa_diamond":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        items = []
        for i, row in enumerate(ds):
            opts = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"],
                    row["Incorrect Answer 3"]]
            opts = [o.strip() for o in opts]
            order = list(range(4))
            random.Random(f"gpqa_diamond:{i}").shuffle(order)
            shuffled = [opts[j] for j in order]
            letter = chr(ord("A") + order.index(0))
            body = "\n".join(f"{chr(ord('A') + k)}) {o}" for k, o in enumerate(shuffled))
            items.append({"id": f"gpqa_diamond_{i}", "answer": letter,
                          "prompt": f"Question: {row['Question'].strip()}\n\n{body}{MCQ_SUFFIX}"})
        return items
    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [{"id": f"gsm8k_{i}", "answer": row["answer"].split("####")[-1].strip().replace(",", ""),
                 "prompt": f"{row['question'].strip()}{GSM_SUFFIX}"} for i, row in enumerate(ds)]
    raise ValueError(name)


def build_requests(spec: CapabilityEvalSpec) -> list[GenRequest]:
    reqs = []
    for bench in spec.benchmarks:
        items = _load(bench)
        if spec.n_items is not None and spec.n_items < len(items):
            items = random.Random(spec.sample_seed).sample(items, spec.n_items)
        reqs += [GenRequest(id=it["id"], messages=[{"role": "user", "content": it["prompt"]}],
                            metadata={"benchmark": bench, "answer": it["answer"]}) for it in items]
    return reqs


def _num(s: str | None) -> float | None:
    if s is None:
        return None
    m = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return float(m[-1]) if m else None


def grade_rows(rows: list[dict], model_label: str) -> list[Record]:
    out = []
    for r in rows:
        md = r["metadata"]
        rec: Record = {"eval": "capability", "item_id": r["id"], "cluster_id": r["id"],
                       "sample_index": r["sample_index"], "model_label": model_label,
                       "benchmark": md["benchmark"], "answer": md["answer"]}
        if r.get("error"):
            out.append({**rec, "error": r["error"]})
            continue
        final = r["final"]
        if md["benchmark"] == "gpqa_diamond":
            m = re.search(r"ANSWER:\s*\(?([A-Da-d])\b", final)
            pred = m.group(1).upper() if m else None
            correct = pred == md["answer"]
        else:
            m = re.search(r"<answer>\s*(.*?)\s*</answer>", final, re.DOTALL)
            pred = m.group(1) if m else None
            pv, av = _num(pred), _num(md["answer"])
            correct = pv is not None and av is not None and abs(pv - av) < 1e-6
        rec.update({"predicted": pred, "correct": int(correct), "final": final,
                    "analysis": r["analysis"], "n_analysis_tokens": r["n_analysis_tokens"],
                    "stop_reason": r["stop_reason"], "parse_ok": r["parse_ok"]})
        out.append(rec)
    return out


def summarize(records: list[Record], n_boot: int, seed: int) -> dict:
    ok = [r for r in records if not r.get("error")]
    m: dict[str, Any] = {}
    for b in sorted({r["benchmark"] for r in ok}):
        rb = [r for r in ok if r["benchmark"] == b]
        m[f"accuracy/{b}"] = cluster_bootstrap(rb, lambda rs: macro_mean(rs, "correct", ()),
                                               n_boot, seed)
        m[f"cot_tokens/{b}"] = length_stats([r["n_analysis_tokens"] for r in rb])
        m[f"truncated_rate/{b}"] = cluster_bootstrap(rb, flag_macro(is_truncated, ()), n_boot, seed)
    return {"eval": "capability", "n_samples": len(records),
            "n_errors": sum(1 for r in records if r.get("error")),
            "n_truncated": sum(1 for r in ok if is_truncated(r)), "metrics": m}
