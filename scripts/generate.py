"""Batch generation (teacher data, carriers). Usage: generate.py CONFIG [--smoke]"""

from cotsl.cli import parse_args, resolve_checkpoint
from cotsl.config import GenerationConfig, load_config
from cotsl.eval.length_style import summarize_rows
from cotsl.gen.runner import GenRequest, generate
from cotsl.io import read_jsonl, write_json, write_jsonl
from cotsl.paths import RESULTS, resolve
from cotsl.provenance import write_run_metadata


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args.config, GenerationConfig, smoke=args.smoke)
    model = resolve_checkpoint(cfg.model)
    prompts = read_jsonl(resolve(cfg.prompts.path))[: cfg.prompts.limit]
    reqs = [GenRequest(id=p["id"], messages=p["messages"], metadata=p.get("metadata", {}))
            for p in prompts]
    out = RESULTS / "gen" / cfg.name
    write_run_metadata(out, cfg, smoke=args.smoke, seeds={"data_gen": cfg.seed},
                       extra={"resolved_checkpoint": model.checkpoint})
    rows, stats = generate(reqs, model=model, sampling=cfg.sampling, seed=cfg.seed,
                           backend=cfg.backend, concurrency=cfg.concurrency,
                           cache_dir=cfg.cache_dir)
    write_jsonl(out / "outputs.jsonl", rows)
    write_json(out / "stats.json", {"generation": stats.__dict__, "length_style": summarize_rows(rows)})
    print(f"{stats}\nwrote {out}/outputs.jsonl")


if __name__ == "__main__":
    main()
