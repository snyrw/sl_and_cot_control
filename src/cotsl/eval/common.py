"""Common results schema + statistics for every eval.

Each eval writes, under ``<output_dir>/<run name>/<eval name>/``:

* ``samples.jsonl`` - one record per (item, sample). Required keys:
  ``eval, item_id, cluster_id, sample_index, model_label`` plus eval-specific grading fields.
  ``cluster_id`` is the unit of independence for bootstrap CIs (e.g. the question, which
  CoT-Control crosses with every mode).
* ``summary.json``  - ``{eval, n_samples, n_errors, metrics: {...}, provenance}`` where each
  metric is ``{"value", "ci_low", "ci_high", "n"}``.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from cotsl.io import write_json, write_jsonl

Record = dict[str, Any]


def mean_or_none(xs: Iterable[float | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None]
    return statistics.fmean(vals) if vals else None


def macro_mean(records: list[Record], field: str, group_keys: tuple[str, ...]) -> float | None:
    """Mean of per-group means (CoT-Control macro-averages over (dataset, mode) pairs)."""
    groups: dict[tuple, list] = defaultdict(list)
    for r in records:
        if r.get(field) is not None:
            groups[tuple(r[k] for k in group_keys)].append(float(r[field]))
    if not groups:
        return None
    return statistics.fmean(statistics.fmean(v) for v in groups.values())


def is_truncated(r: Record) -> bool:
    return r.get("stop_reason") == "length"


def is_empty_cot(r: Record) -> bool:
    return not (r.get("analysis") or "").strip()


def flag_macro(flag: Callable[[Record], bool], group_keys: tuple[str, ...]):
    """Stat fn: macro-averaged rate of a per-record boolean (e.g. truncation)."""
    return lambda rs: macro_mean([dict(r, _flag=int(flag(r))) for r in rs], "_flag", group_keys)


def cluster_bootstrap(records: list[Record], stat: Callable[[list[Record]], float | None],
                      n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
                      cluster_key: str = "cluster_id") -> dict[str, float | int | None]:
    """Point estimate + percentile CI, resampling clusters (not rows) with replacement."""
    point = stat(records)
    clusters: dict[Any, list[Record]] = defaultdict(list)
    for r in records:
        clusters[r[cluster_key]].append(r)
    keys = list(clusters)
    out = {"value": point, "ci_low": None, "ci_high": None, "n": len(records),
           "n_clusters": len(keys)}
    if point is None or len(keys) < 2 or n_boot <= 0:
        return out
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sample = [r for k in rng.choices(keys, k=len(keys)) for r in clusters[k]]
        v = stat(sample)
        if v is not None and not math.isnan(v):
            stats.append(v)
    if stats:
        stats.sort()
        out["ci_low"] = stats[int((alpha / 2) * (len(stats) - 1))]
        out["ci_high"] = stats[int((1 - alpha / 2) * (len(stats) - 1))]
    return out


def length_bin(n_tokens: int | None, edges: list[int]) -> str | None:
    if n_tokens is None:
        return None
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= n_tokens < hi:
            return f"[{lo},{hi})"
    return f">={edges[-1]}"


def length_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0}
    s = sorted(values)

    def pct(p: float) -> float:
        return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]

    return {"n": len(s), "mean": statistics.fmean(s), "p10": pct(0.10), "p25": pct(0.25),
            "p50": pct(0.50), "p75": pct(0.75), "p90": pct(0.90), "p99": pct(0.99),
            "max": s[-1]}


def write_eval_outputs(out_dir: str | Path, records: list[Record], summary: dict) -> None:
    out = Path(out_dir)
    write_jsonl(out / "samples.jsonl", records)
    write_json(out / "summary.json", summary)
