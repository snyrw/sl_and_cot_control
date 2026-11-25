# Verification log

What has been checked against primary sources or code, and what the brief (CLAUDE.md) got
wrong. "Source" = where it was checked. Update this file whenever a [VERIFY] item is resolved.

## Resolved

| Claim in brief | Finding | Source |
|---|---|---|
| METR LoRA rank 32 | **Main METR runs use rank 1** (lr 1e-4, batch 4, 1 epoch, Adam β=(0.9, 0.95), linear LR decay, max_length 8192). The "240 examples" run is also rank 1. | `keing1/cot_controllability` @ 9d2c4ec: `scripts/runs/launch_ft.py`, `overnight_ft_eval_v4.py`. Still to check against the blog. |
| METR trained attention-only? | No. METR used tinker-cookbook's trainer, which always calls `create_lora_training_client(base_model, rank)`: attn + MLP + unembed. | cookbook `supervised/train.py` |
| METR's "900-example QA subset" | 100 questions, proportional across GPQA/HLE/MMLU-Pro (seed 42), keyword-bearing questions only, × 9 modes. Our default reproduces this selection (36/39/25). | METR `evals/runner.py`, `datasets/splits.py` |
| METR eval tweak: "analysis channel" wording | Confirmed. Upstream CoTControl `create_controlled_prompt` already does it for any model name containing "gpt-oss". METR rewrites ReasonIF prompts too. | upstream `prompts.py`; METR `evals/prompts.py` |
| METR eval tweak: synonyms not penalised | Confirmed: `include_synonyms=False`, `check_synonyms=False` by default. | METR `evals/modes.py`, `grading.py` |
| CoT-Control categories | Suppression = word, multiple words, ignore question. Addition = repeat-string, end-with-"safe", meow. Style = lower, upper, alternating. | paper Table 1 |
| CoT-Control score | Binary, macro-averaged over (dataset, type) pairs. Also reports meta-discussion rate (among compliant) and score excluding meta-discussion. | paper §2.4 |
| Judge model | gpt-5-mini (upstream CoTControl and METR), for ignore_question and meta-discussion. | upstream `grade_compliance_csv.py` |
| vLLM harmony/LoRA for gpt-oss | Moot for now: Tinker handles gpt-oss sampling and LoRA. Our harmony rendering matches the official HF chat template exactly (test). | `tests/test_formats_and_datum.py` |
| Attention-only LoRA possible | Tinker exposes `train_attn` / `train_mlp` / `train_unembed`. **Not yet confirmed that the server accepts `train_mlp=False` for gpt-oss MoE**: check on the first smoke run. | tinker `service_client.py` |

## Discrepancies to keep in mind

- **METR's MMLU-Pro prompts had no answer options.** Their loader reads `answer_options`
  only; the MMLU-Pro CSV calls it `options`. Upstream reads both, and so do we. This likely
  affects METR's MMLU-Pro CoT lengths and correctness, not compliance grading.
- **ignore_question triple-check.** Paper footnote: "failure only if all three runs fail"
  (lenient). Upstream code: compliant only if all three calls say T (strict). We default to
  the code (`ignore_question_rule: all_pass`).
- **The CoT-Control repo asks developers to exclude it from training data** (canary
  `cotcontrol:26b5c67b-...`). METR's SFT builder can draw from CoT-Control. Phase 1 must
  decide whether our T_treat recipe uses only the ReasonIF / Multilingual-Thinking source.
- ReasonIF `number_words` limits are per-model in upstream (`number_of_words_reference.json`,
  includes gpt-oss-20b but not Qwen3-8B). METR recalibrated per reasoning effort; their
  calibration file is not in the repo.

## Open

- OpenAI's continuous score: no formula published ("roughly the fraction of the CoT that
  followed the instruction"). Ours is the upstream graders' unthresholded rate (see
  `eval/cotcontrol_continuous.py`), so it is our operationalisation, not theirs.
- "CoT = text minus final 10%" for untagged models: units (characters vs tokens) not yet
  checked in the CoT-Control paper or code. `untagged_cot_split` uses characters.
- METR blog numbers (2.9% → 8.8%, ~950 examples), Multilingual-Thinking × ReasonIF prompt
  construction, edit operations: check in `training/sft_builder.py` / `transforms.py` at
  the start of Phase 1.
- Redwood steering replication, OLMo-3 revision names, RL-Zero-IF reward code, OpenAI
  monitorability eval license: not started (later phases).
- Thomas Read's elicitation prompt is not public; ours (`configs/prompts/elicited_cotcontrol_v1.txt`)
  is written from its description.
