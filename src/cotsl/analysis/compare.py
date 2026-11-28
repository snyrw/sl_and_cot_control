"""Paired comparison of two eval runs on the same items (e.g. T_treat vs base, or treatment
vs control student). Differences get percentile bootstrap CIs from resampling clusters
(questions) jointly in both runs, so item difficulty cancels out.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from cotsl.eval.common import Record, is_empty_cot, is_truncated, macro_mean
from cotsl.io import read_jsonl

Stat = Callable[[list[Record]], float | None]


def _rate(field: str) -> Stat:
    return lambda rs: macro_mean([r for r in rs if r.get(field) is not None], field, ())


def _flag(fn) -> Stat:
    return lambda rs: (sum(bool(fn(r)) for r in rs) / len(rs)) if rs else None


def _mean(field: str) -> Stat:
    return lambda rs: (sum(r[field] for r in rs) / len(rs)) if rs else None


def _macro_continuous(rs):
    return macro_mean([r for r in rs if r.get("continuous") is not None], "continuous",
                      ("dataset", "mode"))


def cotcontrol_metrics(regimes: list[str]) -> dict[str, tuple[Callable[[Record], bool], Stat]]:
    out: dict[str, tuple[Callable[[Record], bool], Stat]] = {}
    for reg in regimes:
        base = lambda r, reg=reg: r["regime"] == reg and r["mode"] != "baseline"  # noqa: E731
        graded = lambda r, b=base: b(r) and r.get("compliant") is not None  # noqa: E731
        out[f"{reg}/compliance_pooled"] = (graded, _rate("compliant"))
        for cat in ("suppression", "addition", "style"):
            out[f"{reg}/compliance_pooled/{cat}"] = (
                lambda r, g=graded, c=cat: g(r) and r["category"] == c, _rate("compliant"))
        out[f"{reg}/continuous_macro"] = (base, _macro_continuous)
        out[f"{reg}/correct"] = (lambda r, b=base: b(r) and r.get("correct") is not None,
                                 _rate("correct"))
        out[f"{reg}/meta_regex_rate"] = (base, _flag(lambda r: r.get("meta_regex")))
        out[f"{reg}/truncated_rate"] = (base, _flag(is_truncated))
        out[f"{reg}/empty_cot_rate"] = (base, _flag(is_empty_cot))
        out[f"{reg}/cot_tokens_mean"] = (lambda r, b=base: b(r) and "n_analysis_tokens" in r,
                                         _mean("n_analysis_tokens"))
    return out


def per_mode_metrics(regimes: list[str], modes: list[str]):
    return {f"{reg}/compliance/{m}": (
        lambda r, reg=reg, m=m: r["regime"] == reg and r["mode"] == m
        and r.get("compliant") is not None, _rate("compliant"))
        for reg in regimes for m in modes}


def reasonif_metrics(types: list[str]):
    g = lambda r: r.get("compliant") is not None  # noqa: E731
    out = {"compliance_pooled": (g, _rate("compliant")),
           "truncated_rate": (lambda r: True, _flag(is_truncated)),
           "cot_tokens_mean": (lambda r: "n_analysis_tokens" in r, _mean("n_analysis_tokens"))}
    for t in types:
        out[f"compliance/{t}"] = (lambda r, t=t: g(r) and r["instruction_type"] == t,
                                  _rate("compliant"))
    return out


def capability_metrics(benches: list[str]):
    return {f"accuracy/{b}": (lambda r, b=b: r["benchmark"] == b, _rate("correct"))
            for b in benches}


def paired_compare(a: list[Record], b: list[Record], metrics: dict, n_boot: int = 2000,
                   seed: int = 0, alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    a = [r for r in a if not r.get("error")]
    b = [r for r in b if not r.get("error")]
    ca: dict[Any, list[Record]] = defaultdict(list)
    cb: dict[Any, list[Record]] = defaultdict(list)
    for r in a:
        ca[r["cluster_id"]].append(r)
    for r in b:
        cb[r["cluster_id"]].append(r)
    keys = sorted(set(ca) & set(cb))
    rng = random.Random(seed)
    boots = [rng.choices(keys, k=len(keys)) for _ in range(n_boot)]
    out = {}
    for name, (sel, stat) in metrics.items():
        def value(clusters, ks):
            return stat([r for k in ks for r in clusters[k] if sel(r)])

        va, vb = value(ca, keys), value(cb, keys)
        if va is None or vb is None:
            continue
        diffs = []
        for ks in boots:
            x, y = value(ca, ks), value(cb, ks)
            if x is not None and y is not None:
                diffs.append(y - x)
        diffs.sort()
        lo = diffs[int(alpha / 2 * (len(diffs) - 1))] if diffs else None
        hi = diffs[int((1 - alpha / 2) * (len(diffs) - 1))] if diffs else None
        out[name] = {"a": va, "b": vb, "diff": vb - va, "ci_low": lo, "ci_high": hi,
                     "p_le_0": (sum(d <= 0 for d in diffs) / len(diffs)) if diffs else None,
                     "n_a": sum(sel(r) for r in a), "n_b": sum(sel(r) for r in b),
                     "n_clusters": len(keys)}
    return out


def compare_runs(run_a: Path, run_b: Path, n_boot: int = 2000, seed: int = 0,
                 per_mode: bool = False) -> dict[str, dict]:
    res = {}
    for ev in ("cotcontrol", "reasonif", "capability"):
        pa, pb = run_a / ev / "samples.jsonl", run_b / ev / "samples.jsonl"
        if not (pa.exists() and pb.exists()):
            continue
        a, b = read_jsonl(pa), read_jsonl(pb)
        if ev == "cotcontrol":
            regimes = sorted({r["regime"] for r in a} & {r["regime"] for r in b})
            metrics = cotcontrol_metrics(regimes)
            if per_mode:
                modes = sorted({r["mode"] for r in a if r["mode"] != "baseline"})
                metrics |= per_mode_metrics(regimes, modes)
        elif ev == "reasonif":
            metrics = reasonif_metrics(sorted({r["instruction_type"] for r in a}))
        else:
            metrics = capability_metrics(sorted({r["benchmark"] for r in a}))
        res[ev] = paired_compare(a, b, metrics, n_boot=n_boot, seed=seed)
    return res
