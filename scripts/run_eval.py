"""Run an eval suite against one model. Usage: run_eval.py CONFIG [--smoke]"""

import json

from cotsl.cli import parse_args, resolve_checkpoint
from cotsl.config import EvalConfig, load_config
from cotsl.eval.run import run_evals


def _headline(summary: dict) -> dict:
    keep = ("compliance_macro", "continuous_macro", "accuracy/", "suppression", "addition",
            "style", "correct/")
    return {k: (round(v["value"], 4) if isinstance(v, dict) and v.get("value") is not None else v)
            for k, v in summary["metrics"].items()
            if any(s in k for s in keep) and "by_length" not in k and "/compliance/" not in k}


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, EvalConfig, smoke=args.smoke)
    cfg = cfg.model_copy(update={"model": resolve_checkpoint(cfg.model)})
    for name, s in run_evals(cfg, smoke=args.smoke).items():
        print(f"== {name}: n={s['n_samples']} errors={s['n_errors']}")
        print(json.dumps(_headline(s), indent=1))


if __name__ == "__main__":
    main()
