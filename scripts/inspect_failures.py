"""How do models fail? Read-only failure-mode breakdown of saved eval samples.

  inspect_failures.py LABEL=RUN_DIR[,RUN_DIR...] [LABEL=...] [--examples N]

e.g. pool the three step-60 seeds against base:

  inspect_failures.py base=results/eval/phase0-eval-gptoss-base \\
      ft60=results/eval/phase1-eval-ttreat-step60,results/eval/phase1-eval-ttreat-seed1-step60,results/eval/phase1-eval-ttreat-seed2-step60

Sections:
  1. CoT-Control case modes (upper/lowercase): never started / started then reverted /
     near-miss slips; word suppression: banned word in the first 10% (restating the task) vs
     later. Per prompt regime, with the rate of CoTs that mention the rule.
  2. ReasonIF failure classes: end phrase missing / misplaced / only in the final answer;
     JSON opened but not valid; capitals started then reverted; comma counts.
  3. Compliance by CoT length (<500 vs >=500 analysis tokens).
  4. Where uppercase CoTs revert (word index), for ReasonIF english_capital.
  5. Accuracy of compliant vs non-compliant CoT-Control samples (is compliance "cheap"?).
  6. (--examples N) N short failure excerpts per class.

Heuristic classes (regexes, letter fractions); the benchmark graders remain the source of
truth for compliance. Nothing here changes results.
"""

from __future__ import annotations

import argparse
import collections
import re
import statistics as st
import textwrap
from pathlib import Path

from cotsl.io import read_jsonl

META = re.compile(r"\b(requirement|constraint|instruction|not allowed|must not|avoid (using )?the "
                  r"word|uppercase|lowercase|all caps|capital letters)\b", re.IGNORECASE)
LONG = 500  # analysis tokens


def _letters(s):
    return [c for c in s if c.isalpha() and c.isascii()]


def _frac(s, upper=True):
    L = _letters(s)
    if not L:
        return None
    return sum((c.isupper() if upper else c.islower()) for c in L) / len(L)


def _banned(prompt):
    m = re.search(r"Do not use the words? (.+?) in your", prompt)
    return re.findall(r"'([^']+)'", m.group(1)) if m else []


def case_class(r, upper=True):
    a = r["analysis"]
    head, whole = _frac(a[:200], upper), _frac(a, upper)
    if head is None:
        return "empty"
    if r.get("compliant"):
        return "compliant"
    if head < 0.5:
        return "never started"
    return "started, then reverted" if whole < 0.9 else "near-miss (slips)"


def supp_class(r):
    if r.get("compliant"):
        return "compliant"
    a, ws = r["analysis"], _banned(r["prompt"])
    if not ws:
        return "fail (word list not parsed)"
    pos = [m.start() for w in ws for m in re.finditer(re.escape(w), a, re.IGNORECASE)]
    if not pos:
        return "fail, word not found (synonym/grader)"
    return "word in first 10% (restating task)" if min(pos) / max(1, len(a)) < 0.1 \
        else "word later in CoT"


def rif_class(r):
    a, f, t = r["analysis"], r.get("final") or "", r["instruction_type"]
    if r.get("compliant"):
        return "compliant"
    if t == "end_checker":
        m = re.search(r'exact phrase "([^"]+)"', r["prompt"])
        p = m.group(1).lower() if m else None
        if p and p in a.lower():
            return "phrase in CoT but not at end"
        if p and p in f.lower():
            return "phrase only in final answer"
        return "phrase missing"
    if t == "json_format":
        s = a.lstrip()
        if s.startswith(("```json", "{")):
            return "opened JSON, not valid (e.g. never closed)"
        if "```json" in a or '{"' in a:
            return "JSON later in CoT"
        return "JSON only in final answer" if ("```json" in f or f.lstrip().startswith("{")) \
            else "no JSON in CoT"
    if t == "english_capital":
        h = _frac(a[:200]) or 0
        if h < 0.5:
            return "never started uppercase"
        return "started uppercase, reverted" if (_frac(a) or 0) < 0.95 else "near-miss"
    if t == "no_comma":
        return "1-2 commas" if a.count(",") <= 2 else "3+ commas"
    return "non-compliant"


def revert_word(a, win=15):
    w = a.split()
    for i in range(max(1, len(w) - win)):
        lo = _frac(" ".join(w[i:i + win]), upper=False)
        if lo is not None and lo > 0.5:
            return i
    return None


def _load(dirs, ev):
    out = []
    for d in dirs:
        p = Path(d) / ev / "samples.jsonl"
        if p.exists():
            out += [r for r in read_jsonl(p) if not r.get("error")]
    return out


def _dist(rows, fn):
    c = collections.Counter(fn(r) for r in rows)
    n = len(rows)
    return ", ".join(f"{k} {100 * v / n:.0f}%" for k, v in c.most_common())


def _rate(rows):
    g = [r for r in rows if r.get("compliant") is not None]
    return f"{100 * sum(bool(r['compliant']) for r in g) / len(g):5.1f}% (n={len(g)})" if g else "   -"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="LABEL=dir[,dir...]")
    p.add_argument("--examples", type=int, default=0)
    a = p.parse_args()
    runs = {s.split("=", 1)[0]: s.split("=", 1)[1].split(",") for s in a.runs}
    cc = {k: _load(v, "cotcontrol") for k, v in runs.items()}
    ri = {k: _load(v, "reasonif") for k, v in runs.items()}

    print("1. CoT-Control failure classes")
    modes = {"uppercase_thinking": lambda r: case_class(r, True),
             "lowercase_thinking": lambda r: case_class(r, False),
             "word_suppression": supp_class, "multiple_word_suppression": supp_class}
    for reg in sorted({r["regime"] for rs in cc.values() for r in rs}):
        print(f"\n  [{reg} prompt]")
        for mode, fn in modes.items():
            print(f"  -- {mode}")
            for k, rs in cc.items():
                xs = [r for r in rs if r["regime"] == reg and r["mode"] == mode]
                if xs:
                    meta = 100 * sum(bool(META.search(r["analysis"])) for r in xs) / len(xs)
                    tok = st.median(r["n_analysis_tokens"] for r in xs)
                    print(f"     {k:8s} n={len(xs):3d} medtok {tok:5.0f} mentions-rule {meta:4.1f}% | "
                          + _dist(xs, fn))

    print("\n2. ReasonIF failure classes")
    for t in ("end_checker", "json_format", "english_capital", "no_comma"):
        print(f"  -- {t}")
        for k, rs in ri.items():
            xs = [r for r in rs if r["instruction_type"] == t]
            if xs:
                print(f"     {k:8s} n={len(xs):3d} | " + _dist(xs, rif_class))

    print(f"\n3. Compliance by CoT length (<{LONG} vs >={LONG} analysis tokens)")
    for k in runs:
        for t in sorted({r["instruction_type"] for r in ri[k]}):
            xs = [r for r in ri[k] if r["instruction_type"] == t]
            print(f"     {k:8s} reasonif/{t:20s} short {_rate([r for r in xs if r['n_analysis_tokens'] < LONG])}"
                  f"   long {_rate([r for r in xs if r['n_analysis_tokens'] >= LONG])}")
        for reg in sorted({r["regime"] for r in cc[k]}):
            xs = [r for r in cc[k] if r["regime"] == reg and r["mode"] != "baseline"]
            print(f"     {k:8s} cotcontrol/{reg:18s} short {_rate([r for r in xs if r['n_analysis_tokens'] < LONG])}"
                  f"   long {_rate([r for r in xs if r['n_analysis_tokens'] >= LONG])}")

    print("\n4. Where uppercase CoTs revert (ReasonIF english_capital, CoTs that start uppercase)")
    for k, rs in ri.items():
        xs = [r for r in rs if r["instruction_type"] == "english_capital"
              and (_frac(r["analysis"][:200]) or 0) > 0.5]
        rev = [w for w in (revert_word(r["analysis"]) for r in xs) if w is not None]
        if rev:
            q = sorted(rev)
            print(f"     {k:8s} started {len(xs)}, reverted {len(rev)}; revert word median {st.median(q)}"
                  f" (IQR {q[len(q) // 4]}–{q[3 * len(q) // 4]})")
        elif xs:
            print(f"     {k:8s} started {len(xs)}, reverted 0")

    print("\n5. Accuracy of compliant vs non-compliant CoT-Control samples")
    for k, rs in cc.items():
        base = [r for r in rs if r["mode"] == "baseline" and r.get("correct") is not None]
        if base:
            print(f"     {k:8s} no-instruction baseline acc "
                  f"{100 * sum(bool(r['correct']) for r in base) / len(base):.1f}%")
        for reg in sorted({r["regime"] for r in rs}):
            xs = [r for r in rs if r["regime"] == reg and r["mode"] != "baseline"
                  and r.get("compliant") is not None and r.get("correct") is not None]
            for lab, ys in (("compliant", [r for r in xs if r["compliant"]]),
                            ("non-compliant", [r for r in xs if not r["compliant"]])):
                if ys:
                    acc = 100 * sum(bool(r["correct"]) for r in ys) / len(ys)
                    print(f"     {k:8s} {reg:9s} {lab:14s} n={len(ys):4d} acc {acc:5.1f}%"
                          f" medtok {st.median(r['n_analysis_tokens'] for r in ys):5.0f}")

    if a.examples:
        print(f"\n6. Examples ({a.examples} per class)")
        for k in runs:
            for t in ("json_format", "english_capital", "end_checker"):
                by = collections.defaultdict(list)
                for r in ri[k]:
                    if r["instruction_type"] == t and not r.get("compliant"):
                        by[rif_class(r)].append(r)
                for cls, xs in by.items():
                    for r in xs[:a.examples]:
                        s = r["analysis"].replace("\n", " ⏎ ")
                        print(f"\n  [{k} | reasonif/{t} | {cls} | {r['n_analysis_tokens']} tok]\n    "
                              + textwrap.shorten(s[:300], 300) + "\n    … "
                              + textwrap.shorten(s[-200:], 200))


if __name__ == "__main__":
    main()
