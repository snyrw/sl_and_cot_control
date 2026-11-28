# cotsl: CoT controllability × subliminal learning

Project brief: [CLAUDE.md](CLAUDE.md). Verification log: [docs/VERIFY.md](docs/VERIFY.md).
Stage-by-stage audit of T_treat vs METR's blog run: [docs/PIPELINE_VS_METR.md](docs/PIPELINE_VS_METR.md).
Deferred GPU backend (OLMo-3, steering): [docs/DEFERRED.md](docs/DEFERRED.md).

## Setup

```bash
git submodule update --init --recursive   # CoTControl + ReasonIF (pinned, same SHAs as METR)
uv sync
cp .env.example .env                      # TINKER_API_KEY, OPENAI_API_KEY, HF_TOKEN (never commit)
uv run pytest -q                          # offline tests, no API keys needed
```

GPQA-Diamond is gated on HF: accept its terms and set `HF_TOKEN`.

## Running

Every script takes one YAML config plus an optional `--smoke`, which applies the config's
`smoke:` block (a few examples, a few steps):

```bash
uv run python scripts/generate.py   configs/phase0/gen_gptoss_base.yaml --smoke
uv run python scripts/train_sft.py  configs/phase0/sft_gptoss.yaml --smoke
uv run python scripts/run_eval.py   configs/phase0/eval_gptoss_base.yaml --smoke
uv run python scripts/phase0_pipeline.py configs/phase0/pipeline.yaml --smoke   # all of the above
```

## Layout

```
src/cotsl/
  config.py       pydantic schemas for every run kind; YAML loading (_base_, smoke:)
  provenance.py   run_meta.json: resolved config, git SHA + dirty files, submodules, seeds, versions
  gen/            formats.py (harmony / Qwen3 render+parse+thinking spans), backend.py (Tinker;
                  vllm deferred), cache.py (content-addressed, never overwrites), runner.py
  train/          datum.py (assistant-only loss; analysis_mode train|context_only|drop), sft.py
  eval/           cotcontrol (+ _continuous), reasonif, capability, length_style, judge, common
configs/          one YAML per run; prompts/ for eval/elicitation prompts
data/prompts/     small committed prompt sets; data/generated/ + data/sft/ are gitignored
results/          per-run outputs (gitignored)
```

## Conventions

- **Seeds:** `seed` in generation configs (data generation), SFT configs (LoRA init + data
  order), and eval configs (eval sampling) are separate. Each item's sampling seed is derived
  from (seed, rendered-prompt hash, sample index), so it doesn't depend on order or subset.
- **Cache:** `data/generated/<model>-<key>/`. The key covers model identity (base, checkpoint,
  revision, steering), backend, sampling params and seed. Items are keyed by rendered prompt
  tokens, so the template, pinned date and system prompt all count. Nothing is regenerated
  silently; delete the directory to regenerate.
- **gpt-oss:** harmony via tinker-cookbook's renderer, with the system-message date pinned
  (`current_date`) and explicit `reasoning_effort`. A `system` role message becomes the
  harmony developer message, which matches the official chat template. Outputs keep both
  channels (`analysis`, `final`) and the raw completion tokens.
- **W&B (training runs):** set `wandb.enabled: true` in an SFT config (the Phase 0/1 configs
  already do). Real runs log to project `cotsl` (grouped by phase), smoke runs to `cotsl-smoke`.
  Runs go to the API key's default entity unless `wandb.entity` is set. Each run logs the
  resolved config, git info, LoRA settings, data stats, per-step NLL/LR/tokens and the
  Tinker checkpoint paths, and writes its URL to `results/sft/<run>/wandb_run.json`. The
  local files stay the source of truth; W&B only mirrors them.
- **Checkpoints in configs:** `model.checkpoint: "sft:<run name>"` resolves to that SFT run's
  final Tinker sampler path at run time.
- **Eval outputs:** `results/eval/<run>/<eval>/samples.jsonl` (one record per sample) and
  `summary.json` (metrics with percentile bootstrap CIs over questions, `cluster_id`).
  Treatment-vs-control comparisons across seeds go in `analysis/` (Phase 1+).
