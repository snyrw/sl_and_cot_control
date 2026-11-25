"""Phase 0 end-to-end check. Usage: phase0_pipeline.py configs/phase0/pipeline.yaml [--smoke]

generate (teacher) -> SFT rows (data/sft/<gen name>.jsonl) -> LoRA SFT -> eval the student.
--smoke is forwarded to every stage, so the whole chain runs in minutes.
"""

from __future__ import annotations

from cotsl.cli import parse_args, resolve_checkpoint
from cotsl.config import (
    EvalConfig,
    GenerationConfig,
    SFTConfig,
    StrictModel,
    load_config,
    load_raw_config,
)
from cotsl.eval.run import run_evals
from cotsl.gen.runner import GenRequest, generate
from cotsl.io import read_jsonl, write_jsonl
from cotsl.paths import REPO_ROOT, resolve
from cotsl.train.datum import gen_rows_to_sft_rows
from cotsl.train.sft import train_sft


class SFTRowsSpec(StrictModel):
    strip_system: bool = False
    keep_analysis: bool = True


class PipelineConfig(StrictModel):
    name: str
    generation: str
    sft_rows: SFTRowsSpec = SFTRowsSpec()
    sft: str
    eval: str


def main() -> None:
    args = parse_args(__doc__)
    pipe = PipelineConfig.model_validate(load_raw_config(args.config))
    gen = load_config(pipe.generation, GenerationConfig, smoke=args.smoke)
    sft = load_config(pipe.sft, SFTConfig, smoke=args.smoke)
    ev = load_config(pipe.eval, EvalConfig, smoke=args.smoke)

    # 1. teacher generations (cached)
    prompts = read_jsonl(resolve(gen.prompts.path))[: gen.prompts.limit]
    rows, stats = generate(
        [GenRequest(p["id"], p["messages"], p.get("metadata", {})) for p in prompts],
        model=resolve_checkpoint(gen.model), sampling=gen.sampling, seed=gen.seed,
        backend=gen.backend, concurrency=gen.concurrency, cache_dir=gen.cache_dir)
    print("generation:", stats)

    # 2. SFT rows
    sft_rows = gen_rows_to_sft_rows(rows, **pipe.sft_rows.model_dump())
    expected = resolve(sft.data.path)
    write_jsonl(expected, sft_rows)
    print(f"wrote {len(sft_rows)} SFT rows -> {expected.relative_to(REPO_ROOT)}")

    # 3. train
    print("sft:", train_sft(sft, smoke=args.smoke))

    # 4. eval the student
    ev = ev.model_copy(update={"model": resolve_checkpoint(ev.model)})
    for name, s in run_evals(ev, smoke=args.smoke).items():
        print(f"eval {name}: n={s['n_samples']} errors={s['n_errors']}")


if __name__ == "__main__":
    main()
