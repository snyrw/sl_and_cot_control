# T_treat training pipeline vs METR's blog run: stage-by-stage audit

This compares our T_treat pipeline with METR's blog run, stage by stage. Our side is checked
against our code and run records. METR's side is checked against `keing1/cot_controllability`
@ 62c786d (2026-04-01, `scripts/runs/generate_sft_data.py`, `scripts/runs/ft_eval_stripped.py`,
`src/controllability/...`) and the library versions that script requires. Audited 2026-11-27.

**Status key:**
- **Verified identical**: checked programmatically, or line by line in code.
- **Matched by design**: same setting, but the outputs are stochastic or external.
- **Deliberate deviation**: we chose to differ, and the choice is documented.
- **Unknown**: METR didn't publish it, or it can't be checked.

## 0. Library versions (a hidden dependency)

| | METR blog run | Ours |
|---|---|---|
| tinker-cookbook | **unpinned**, but `ft_eval_stripped.py` imports `supervised.common.datum_from_tokens_weights`, removed from the cookbook on 2025-12-11 (944af9a). It must have been a version from **before 2025-12-11**, roughly 4 months older than the run. | 0.7.1 |
| tinker SDK | unpinned; the version paired with that cookbook | 0.40.0 |
| Client-side API defaults | SDK @ 2025-12-08: `create_lora_training_client(rank=32, seed=None, train_mlp=True, train_attn=True, train_unembed=True)`; `AdamParams` defaults lr 1e-4, β=(0.9, 0.95), eps 1e-12, weight_decay 0, grad_clip_norm 0 | identical signature and defaults (0.40 adds an `optimizer=` argument that defaults to AdamW; `optim_step(AdamParams)` still governs) |
| Tinker **server** | April 2026 | Nov 2026. **Unknown** whether training or sampling numerics changed. The SDK changelogs document API changes only. |

## 1. Stage-1 rollouts (base model answers with the instruction stripped)

| Item | METR | Ours | Status |
|---|---|---|---|
| Question pool | `external/reasonIF_ft/train-*.parquet`, 953 rows | same file (blob `25ebc2a0…`, sha256 `be96d045…`) | verified identical |
| Generation prompt (instruction stripped) | `strip_reasoning_instruction(extract_user_prompt(...))` | ours; checked by running METR's function on all 953 rows | **verified identical (953/953)** |
| Backend / model | Tinker `openai/gpt-oss-20b` via `TinkerClient` | Tinker `openai/gpt-oss-20b` | matched |
| Renderer | pre-Dec-2025 `GptOssRenderer(use_system_prompt=True, reasoning_effort=X)` | cookbook 0.7.1 `GptOssRenderer`, same system-prompt text | matched |
| System-prompt date | `datetime.now()` at generation time (~late March 2026) | pinned `2026-01-01` | trivial difference |
| **Reasoning effort** | `--reasoning-effort` flag; if omitted, falls back to `"none"` (default changed from `"high"` to `"none"` on 2026-03-26). Not recorded. | `medium` | **unknown** (see VERIFY.md; judged unlikely to be `none`) |
| Sampling | temp 1.0, max_tokens 16384, no top_p/top_k, no seed | temp 1.0, max_tokens 16384, top_p 1, top_k -1 (no-ops), per-item derived seed | matched (seeding changes reproducibility, not the distribution) |
| Samples per row | one rollout per row; repeated questions get separate samples | same (`sample_offset` for the 53 repeated questions) | matched |
| Parsing | regex on the decoded harmony text; `.strip()` on CoT and final; if no final channel, the final becomes the CoT with markers stripped | token-level channel parse | equivalent on our data: all 953 rollouts stopped normally, none truncated, no stray whitespace, 1 empty final |
| Rollout content | not released | ours | **unknown** whether the distributions match |

## 2. Stage-2 edits

| Item | METR | Ours | Status |
|---|---|---|---|
| english_capital / no_comma / end_checker / json_format | `transforms.py` (`.upper()`, remove `,`, `rstrip` + `\n` + phrase if absent, `` ```json\n{"reasoning": json.dumps(cot)}\n``` ``) | ours; checked by running METR's functions on our raw CoTs | **verified byte-identical (656/656 rows)** |
| number_words | limit = `max(int(0.7·\w+ count), 50)`; GPT-4.1 condenses to `max(int(0.85·limit), 30)`; hard-truncate to limit−1 if still over; rewrite "less than N words" in the prompt | same rule, same editor model and prompt | limits verified (143/143); condensation text stochastic |
| reasoning_language | GPT-4.1, blog prompt ("Translate… Output ONLY the translated text") | revised prompt (METR 9d2c4ec) + GPT-5.4 | **deliberate deviation** (52/134 blog-setup outputs were severe failures). Tested: the blog-data run's CoT-Control score (2.8%) lies inside our seed range. |
| Retries / non-compliant rows | up to 3 retries; non-compliant rows kept; only errors dropped | same | matched |
| Training prompt | original instruction, "reasoning" → "analysis channel", word limit rewritten | same; checked against METR's functions | **verified identical (953/953**, including the 139 rows where the word limit changed) |
| Rows in final set | 953 if no errors (not released) | 953 | unknown for METR |
| Row order | pool index order (`stage2` iterates `row_indices`; errors dropped) | pool index order | identical if METR had no errors |

## 3. Tokenisation and loss masking

| Item | METR | Ours | Status |
|---|---|---|---|
| Rendering | old `GptOssRenderer._render_message`: system prompt + `<\|start\|>user<\|message\|>…<\|end\|>` + `<\|start\|>assistant` + action `<\|channel\|>analysis<\|message\|>{CoT}<\|end\|><\|start\|>assistant<\|channel\|>final<\|message\|>{final}<\|return\|>` | cookbook 0.7.1 renderer | **verified identical tokens and weights on all 953 rows** (with the same date) |
| What gets loss | action part of the last assistant message (`TrainOnWhat.ALL_ASSISTANT_MESSAGES`; single-turn, so the same as LAST): CoT, the second `<\|start\|>assistant` header, final answer, `<\|return\|>`. Prompt and system: 0. | `LAST_ASSISTANT_MESSAGE`, `analysis_mode: train` | verified identical |
| Datum | `datum_from_tokens_weights`: truncate to 8192 from the right, shift by one, weights 0/1 | `datum_from_model_input_weights(reduction="none")`: same truncation and shift, raw weights | matched |
| Overlength | 1 row (`metr_pool_901`, 8680 tokens) loses its tail, including the final answer and `<\|return\|>` | same | matched |
| Loss | `cross_entropy`, **sum** over weighted tokens (no per-example or per-batch normalisation) | same | matched |

## 4. Optimisation

| Item | METR (`ft_eval_stripped.py` Config + old `supervised/train.py`) | Ours (run_meta.json + metrics.jsonl) | Status |
|---|---|---|---|
| LoRA rank | 32 | 32 | verified |
| Adapted modules | client defaults: attn + MLP (incl. MoE experts) + unembed | explicit attn + MLP + unembed | verified |
| LoRA init seed | `None` (random, server-side) | `seed=0` | difference in the noise only |
| Optimizer | Adam(W), lr 1e-4, β=(0.9, 0.95), eps 1e-8, wd 0, no clipping | same | verified |
| LR schedule | linear `1 − step/238` | same; logged LR at step 60 = 7.478991596638657e-05, identical to METR's logged value | verified |
| Batch / epochs / steps | 4 / 1 / 238 (953 // 4; the last shuffled row is dropped) | same; 952 distinct examples seen | verified |
| **Data order** | `set_epoch(seed=0)` → `random.Random(0).shuffle(range(953))` | `random.Random(0).shuffle(order)` | **identical permutation: our seed-0 batches are METR's batches**, if their parquet had all 953 rows in pool order. Seeds 1 and 2 change both the order and the LoRA init. |
| Pipelining | async: forward/backward then optim per step, submitted in order | synchronous, same order | equivalent |
| Evaluators during training | `eval_every=10`, but no evaluators registered, so nothing runs | none | matched |
| Checkpoints | `save_every=60`: saves `000060` **after the step-60 update (61 updates)**, plus `final` | saves `step00060` after 60 updates, plus `final` (238) | trivial difference (one batch of 4) |
| Logged metrics | step, lr, NLL, token counts, timing | same, plus optimizer metrics; **forward/backward metrics (MoE routing balance) added 2026-11-27**, not present in runs before that | logging only |

## 5. MoE and gpt-oss specifics on Tinker

These are inherent to Tinker and apply to both runs. As far as can be checked, nothing here
differs between METR's run and ours.

- **Model:** gpt-oss-20b is a MoE (21B total / ~3.6B active parameters, 24 layers, 32 experts,
  top-4 routing, hidden size 2880). Its released expert weights are MXFP4. The cookbook's
  export table lists GPT-OSS on Tinker as **bf16**, i.e. training runs on dequantised weights.
  That export table also notes "serving blocked by mxfp4+LoRA incompatibility" for *external*
  serving of merged adapters. This does not affect Tinker-internal sampling, which both METR
  and we use.
- **Expert LoRA scheme:** shared-outer. The LoRA factor touching the hidden dimension is
  shared across experts, and the other factor is expert-specific. This is documented in the
  cookbook's `get_lora_param_count`, and was already the default (`shared_expert_outer_loras=True`)
  in the April-2026 cookbook. So METR's run very likely used the same scheme.
- **LoRA size:** at rank 32 with attn + MLP + unembed: 241,424,384 trainable parameters
  (7,544,512 per rank).
- **Not exposed or documented:**
  - whether the router is adapted
  - the LoRA alpha / scaling (the cookbook says `lora_alpha` is "currently unused")
  - the LoRA init distribution
  - compute precision
  - **We can't tell whether any of these changed on the server between April and November.**
- **Expert routing balance** (`e_frac_with_tokens`, `e_max_violation`, …) is reported in
  forward/backward metrics. We now log it. Neither METR's logs nor our earlier runs have it.
- **The brief's "attention-only" default for gpt-oss** applies to *students* (Phase 2/3). T_treat
  follows METR (all modules).

## 6. Evaluation (for completeness; base-model results match METR's)

| Item | METR | Ours | Status |
|---|---|---|---|
| CoT-Control | 100 QA questions × 9 modes + baseline, METR keyword filter, `seed=42`, temp 1.0, max_tokens 28000, medium effort, "analysis channel" wording, synonyms not penalised | same | matched (base 1.7 vs METR 2.1) |
| ReasonIF | all 300, upstream checkers, word limits from `number_of_words_reference.json` | same | matched (base 16.3 vs ~15.0) |
| Grading conventions | see VERIFY.md (empty CoT, judge cap, ignore_question, MMLU-Pro options) | replicated | matched |
| FT checkpoint sampled | sampler weights from `save_checkpoint(kind="both")` | `save_weights_for_sampler` | matched |
| Eval date in system prompt | `datetime.now()` at eval time | `2026-01-01` | trivial difference |

## 7. What METR did not release (cannot be audited)

- The blog-run SFT parquet (`gpt-oss-20b-reasonif-sft-stripped-ac.parquet`), its rollouts,
  edit outputs and final row count.
- The stage-1 `--reasoning-effort` value.
- Installed library versions (only constrained: cookbook before 2025-12-11).
- Tinker run IDs, checkpoints and adapter weights; training metrics for the blog run (only a
  March run with an older recipe is committed).
- Eval outputs (per-sample rollouts) for the blog run; only charts. Training-seed count (likely 1).
- The plotting script for the blog figures.

## 8. What is left to explain the gap (7.3% vs our 1.8–3.1%)

Ranked by how much the evidence allows:

1. **The rollouts themselves.** They're different samples, possibly at a different effort,
   from the same model, sampled ~7 months apart on Tinker. Unauditable without METR's data.
2. **Server-side Tinker changes** (LoRA implementation, numerics) between April and November.
   Unauditable. A higher LR did not reproduce METR's faster in-distribution learning, so
   these would have to be more than an LR rescaling.
3. **A high draw from LoRA-init variance on METR's side.** Our seed-0 run shares METR's exact
   data order, so the remaining run-level noise is the LoRA init (plus the data in item 1). Our
   three seeds span only 1.3 points.

Already ruled out or verified:
- hyperparameters, schedule and optimizer defaults
- rendering, masking, truncation and loss reduction
- prompts, deterministic edits and word limits
- data order
- the translation deviation
- learning rate as a speed-of-learning explanation
- eval pipeline (the base models match)
