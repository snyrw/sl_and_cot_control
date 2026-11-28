"""Paired comparison of two eval runs (read-only; does not change results).

  compare_evals.py results/eval/<A> results/eval/<B> [--per-mode]

Prints B − A for every metric with a 95% bootstrap CI over questions (paired), and writes
results/compare/<A>__vs__<B>.json.
"""

import argparse
from pathlib import Path

from cotsl.analysis.compare import compare_runs
from cotsl.io import write_json
from cotsl.paths import RESULTS


def _pct(v):
    return "   -  " if v is None else f"{100 * v:6.1f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--per-mode", action="store_true")
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()
    ra, rb = Path(a.run_a), Path(a.run_b)
    res = compare_runs(ra, rb, n_boot=a.n_boot, per_mode=a.per_mode)
    print(f"A = {ra.name}\nB = {rb.name}\n(rates in %, cot_tokens in tokens; CI = 95% paired bootstrap over questions)")
    for ev, metrics in res.items():
        print(f"\n== {ev}")
        print(f"  {'metric':<44} {'A':>7} {'B':>7} {'B−A':>7}   95% CI           ")
        for k, m in metrics.items():
            if "tokens" in k:
                print(f"  {k:<44} {m['a']:7.0f} {m['b']:7.0f} {m['diff']:+7.0f}   "
                      f"[{m['ci_low']:+.0f}, {m['ci_high']:+.0f}]")
                continue
            sig = "*" if m["ci_low"] is not None and (m["ci_low"] > 0 or m["ci_high"] < 0) else " "
            print(f"  {k:<44} {_pct(m['a'])}  {_pct(m['b'])}  {100 * m['diff']:+6.1f}{sig}  "
                  f"[{_pct(m['ci_low']).strip()}, {_pct(m['ci_high']).strip()}]")
    out = RESULTS / "compare" / f"{ra.name}__vs__{rb.name}.json"
    write_json(out, res)
    print(f"\n* = 95% CI excludes 0.  Written: {out}")


if __name__ == "__main__":
    main()
