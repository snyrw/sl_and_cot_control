"""Read-only viewer for run outputs. Does not change results.

  inspect_run.py results/eval/<run>                       # headline metrics for every eval
  inspect_run.py results/eval/<run> --all                 # every metric incl. per-mode / length bins
  inspect_run.py results/eval/<run> --samples cotcontrol  # one line per graded sample
  inspect_run.py results/eval/<run> --show 'hle_58|uppercase_thinking|elicited'  # full CoT + answer
  inspect_run.py results/sft/<run>                        # checkpoints, data stats, loss per step
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return "-" if v is None else str(v)


def show_eval(root: Path, show_all: bool) -> None:
    meta = json.loads((root / "run_meta.json").read_text())
    m = meta["config"]["model"]
    print(f"model: {m['base_model']}  checkpoint: {m.get('checkpoint') or '(base)'}  "
          f"effort: {m.get('reasoning_effort')}  git: {(meta['git']['sha'] or '')[:8]}"
          f"{' (dirty)' if meta['git']['dirty'] else ''}  smoke: {meta['smoke']}")
    for summary_path in sorted(root.glob("*/summary.json")):
        s = json.loads(summary_path.read_text())
        g = s.get("generation", {})
        print(f"\n== {s['eval']}  samples={s['n_samples']} errors={s['n_errors']} "
              f"truncated={s.get('n_truncated', '-')} cache_hits={g.get('cache_hits')}/{g.get('requested')}")
        for k, v in s["metrics"].items():
            if not isinstance(v, dict) or "value" not in v:
                continue
            if not show_all and ("by_length" in k or k.count("/") > 2
                                 or ("/compliance/" in k or "/correct/" in k or "/continuous/" in k)):
                continue
            print(f"  {k:<48} {_fmt(v['value']):>6}  CI [{_fmt(v['ci_low'])}, {_fmt(v['ci_high'])}]  "
                  f"n={v['n']} clusters={v.get('n_clusters')}")
        tok = s.get("length_style", {}).get("analysis_tokens", {})
        if tok.get("n"):
            print(f"  CoT tokens: mean {tok['mean']:.0f}, p50 {tok['p50']}, p90 {tok['p90']}, max {tok['max']}")


def show_samples(root: Path, ev: str) -> None:
    for line in (root / ev / "samples.jsonl").open():
        r = json.loads(line)
        if r.get("error"):
            print(f"{r['item_id']:<45} ERROR {r['error']}")
            continue
        fields = {k: r.get(k) for k in ("stop_reason", "n_analysis_tokens", "compliant",
                                        "continuous", "meta_judge", "predicted",
                                        "correct_letter", "correct", "answer_match") if k in r}
        print(f"{r['item_id']:<45} " + " ".join(f"{k}={_fmt(v)}" for k, v in fields.items()))


def show_one(root: Path, item_id: str) -> None:
    for path in root.glob("*/samples.jsonl"):
        for row in path.open():
            r = json.loads(row)
            if r["item_id"] == item_id:
                print(f"--- PROMPT ---\n{r.get('prompt')}\n\n--- COT (analysis) ---\n{r.get('analysis')}"
                      f"\n\n--- FINAL ---\n{r.get('final')}\n\n--- GRADES ---")
                print({k: v for k, v in r.items() if k not in ("prompt", "analysis", "final")})
                return
    print(f"no sample with item_id {item_id!r}")


def show_sft(root: Path) -> None:
    for line in (root / "checkpoints.jsonl").open():
        print("checkpoint:", json.loads(line))
    stats = json.loads((root / "data_stats.json").read_text())
    print("data:", {k: v for k, v in stats.items() if k != "dropped_ids"})
    meta = json.loads((root / "run_meta.json").read_text())
    print("lora:", meta.get("lora_resolved"), " seed:", meta["seeds"])
    for line in (root / "metrics.jsonl").open():
        m = json.loads(line)
        print(f"  step {m['step']:>4}  lr {m['lr']:.2e}  nll {m['train_mean_nll']:.4f}  "
              f"loss_tokens {m['n_loss_tokens']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--all", action="store_true", help="show every metric")
    p.add_argument("--samples", metavar="EVAL", help="list graded samples for one eval")
    p.add_argument("--show", metavar="ITEM_ID", help="print one sample in full")
    a = p.parse_args()
    root = Path(a.run_dir)
    if (root / "checkpoints.jsonl").exists():
        show_sft(root)
    elif a.show:
        show_one(root, a.show)
    elif a.samples:
        show_samples(root, a.samples)
    else:
        show_eval(root, a.all)


if __name__ == "__main__":
    main()
