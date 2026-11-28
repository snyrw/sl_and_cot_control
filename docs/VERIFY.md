# Verification log

What has been checked against primary sources or code, and what the brief (CLAUDE.md) got
wrong. "Source" = where it was checked. Update this file whenever a [VERIFY] item is resolved.

## Resolved

| Claim in brief | Finding | Source |
|---|---|---|
| METR LoRA rank 32 | **Correct for the blog run.** The blog-era script (commit 62c786d, 2026-04-01) uses rank 32, lr 1e-4, batch 4, 1 epoch, max_length 8192, save_every 60, Adam β=(0.9, 0.95), linear decay. Rank 1 comes from *later* experiments (`overnight_ft_eval_v4.py`, `launch_ft.py`, Apr 7+). An earlier entry here said rank 1; that was wrong. | `scripts/runs/ft_eval_stripped.py` @ 62c786d |
| Four models | gpt-oss-20b and 120b (analysis-channel wording, medium effort), Qwen3-8B and 32B. | same |
| ~950 examples / "240" | The question pool has **953 rows**, all from `HuggingFaceH4/Multilingual-Thinking`, each paired with one of the 6 ReasonIF instructions (no_comma 180, end_checker 175, json_format 156, reasoning_language 154, english_capital 145, number_words 143). 953 / batch 4 ≈ 238 steps per epoch, with checkpoints at 60/120/180/final. **"240" = the first checkpoint (step 60 × batch 4 = 240 samples), and the blog reports that checkpoint by default** (figure files `lr1e-4_step60`). An earlier entry here called 240 the total step count; that was wrong. | `external/reasonIF_ft/train-*.parquet` |
| Edit recipe | Generate *without* the instruction (`--strip-instruction`), at temperature 1.0 and max_tokens 16384. Then edit: english_capital = `.upper()`; no_comma = drop commas; end_checker = append the phrase; json_format = wrap as ```` ```json {"reasoning": …}``` ````; reasoning_language = GPT-4.1 translation (temperature 0.3); number_words = **per-example limit = 70% of the rollout's own word count (min 50)**, GPT-4.1 asked to condense to 85% of that, hard truncation if still over, and the prompt's limit rewritten to match. Up to 3 retries. **Only the 6 ReasonIF edits; no CoT-Control data in the blog run.** | `transforms.py`, `generate_sft_data.py` @ 62c786d |
| Editor model | **`openai/gpt-4.1` via OpenRouter** in the blog run. GPT-5.4 is a later default. | same |
| What gets trained | The `messages` column: the prompt *with* the instruction (analysis-channel wording) + edited CoT + the original final answer. Loss on all assistant messages, CoT included. **Rows still non-compliant after 3 retries are kept**; only generation/edit errors are dropped. The parquet also stores `messages_original` (same prompt, *unedited* CoT). | `generate_sft_data.py`, `ft_eval_stripped.py` @ 62c786d |
| Blog-era eval | CoT-Control 100 questions × all 10 modes, ReasonIF all 300, max_tokens 28000, temperature 1.0, seed 42. Identical to our `eval_gptoss_base.yaml`. | `ft_eval_stripped.py` @ 62c786d |
| METR trained attention-only? | No. METR used tinker-cookbook's trainer, which always calls `create_lora_training_client(base_model, rank)`: attn + MLP + unembed. | cookbook `supervised/train.py` |
| METR's "900-example QA subset" | 100 questions, proportional across GPQA/HLE/MMLU-Pro (seed 42), keyword-bearing questions only, × 9 modes. Our default reproduces this selection (36/39/25). | METR `evals/runner.py`, `datasets/splits.py` |
| METR eval tweak: "analysis channel" wording | Confirmed. Upstream CoTControl `create_controlled_prompt` already does it for any model name containing "gpt-oss". METR rewrites ReasonIF prompts too. | upstream `prompts.py`; METR `evals/prompts.py` |
| METR eval tweak: synonyms not penalised | Confirmed: `include_synonyms=False`, `check_synonyms=False` by default. | METR `evals/modes.py`, `grading.py` |
| CoT-Control categories | Suppression = word, multiple words, ignore question. Addition = repeat-string, end-with-"safe", meow. Style = lower, upper, alternating. | paper Table 1 |
| CoT-Control score | Binary, macro-averaged over (dataset, type) pairs. Also reports meta-discussion rate (among compliant) and score excluding meta-discussion. | paper §2.4 |
| Judge model | gpt-5-mini (upstream CoTControl and METR), for ignore_question and meta-discussion. | upstream `grade_compliance_csv.py` |
| vLLM harmony/LoRA for gpt-oss | Moot for now: Tinker handles gpt-oss sampling and LoRA. Our harmony rendering matches the official HF chat template exactly (test). | `tests/test_formats_and_datum.py` |
| Attention-only LoRA possible | Tinker exposes `train_attn` / `train_mlp` / `train_unembed`. **Confirmed 2026-11-25:** the server accepts attention-only (`train_mlp=False, train_unembed=False`) for gpt-oss-20b. The smoke SFT trained, checkpointed and sampled. | tinker `service_client.py`; `results/sft/phase0-sft-gptoss-smoke` |

## Discrepancies to keep in mind

- **METR's MMLU-Pro prompts had no answer options.** Their loader reads `answer_options`
  only; the MMLU-Pro CSV calls it `options`. Upstream reads both, and so do we. Almost
  certainly accidental:
  - Nothing in METR's repo mentions it.
  - Their eval prompt builder deliberately formats choices plus the 'ANSWER: X' line whenever
    a sample has options.
  - Upstream's runner reads both column names, i.e. the CSVs are known to be inconsistent.

  Effect: those 25/100 questions were open-ended with no answer-format line, so they are
  almost never graded correct. METR's accuracy curves pool them, which gives a roughly
  constant offset for base and FT alike. Compliance grading is unaffected, but the CoTs
  answer a different prompt. (METR's `mcq_format=False` in SFT prompts is separate and
  deliberate.) Decided 2026-11-26: we show choices by default. Set
  `mmlu_pro_options: metr_drop` to reproduce METR's prompts for a replication check.
- **Grading follows METR by default** (decided 2026-11-26: results will be compared
  directly with theirs). Changes from upstream behaviour, each a config switch:
  - Headline `compliance_pooled` = compliant / graded, pooled over modes and datasets
    (METR's `overall_rate` in `plot_summary_2026_05_04.py`). Per-mode compliance is pooled
    over datasets (METR `compute_metrics`). The paper's macro-average remains as
    `compliance_macro`. **Not yet confirmed which plotting script produced the blog figure.**
  - Empty CoT: ungraded/excluded (`empty_cot: exclude`), except ignore_question = fail.
  - Judge: `max_completion_tokens=500`, empty/unparseable reply = F, and a judge exception
    leaves the sample ungraded.
  - Already identical: suppression keywords (checked on all 1,169 questions), no synonyms,
    triple-check all_pass, gpt-5-mini, truncated graded as-is, regex meta-discussion.
- **METR's 500-token judge cap empties most of gpt-5-mini's replies.** Reasoning tokens use
  up the budget, the call ends with `finish_reason=length` and empty content, and METR's
  parser turns that into F.
  - Full base gpt-oss-20b eval (2026-11-26): 170/205 (83%) capped ignore_question calls were
    empty; the uncapped judge had none.
  - Base scores were identical anyway (2/200 compliant either way), because base almost never
    complies. The cap should only bite once compliance rises (T_treat, elicited prompt).
  - Decided: the capped judge drives the METR-comparable headline, and the uncapped judge is
    reported as the `@iq_uncapped` secondary.
  - The same cap emptied 32/124 meta-discussion calls. Meta-discussion judging is upstream's
    (METR uses a regex only), so it now always runs uncapped.
- **Truncated CoTs** (hit max_tokens): graded as-is by upstream CoT-Control (no
  finish_reason check, 25k reasoning budget), upstream ReasonIF (no `</think>` → whole response
  treated as reasoning, 16k budget) and METR (28k budget). We do the same, and report
  `truncated_rate` per regime and category so arms can be compared. Exclusions not applied
  (decided 2026-11-26: stay close to METR for now).
- **Empty CoTs.** Upstream CoT-Control and ReasonIF count an empty CoT as non-compliant, and so
  do we. METR instead labels it an error and drops it from the denominator, except for
  ignore_question, where it counts as a fail. Their reason: over OpenRouter, timeouts often come
  back as empty responses with no error ("stealth timeouts", see
  `scripts/runs/rerun_elevated_errors.py`), and they rerun those. Our Tinker path raises and
  retries on failure, so an empty CoT here is more likely real behaviour.
  `empty_cot_rate` is reported.
- **ignore_question triple-check.** Paper footnote: "failure only if all three runs fail"
  (lenient). Upstream code: compliant only if all three calls say T (strict). We default to
  the code (`ignore_question_rule: all_pass`).
- **The CoT-Control repo asks developers to exclude it from training data** (canary
  `cotcontrol:26b5c67b-...`). METR's *later* SFT builder can draw from CoT-Control, but the
  blog run used ReasonIF/Multilingual-Thinking only, so replicating the blog avoids the issue.
- ReasonIF `number_words` limits are per-model in upstream (`number_of_words_reference.json`,
  includes gpt-oss-20b but not Qwen3-8B). METR recalibrated per reasoning effort; their
  calibration file is not in the repo.

## Open

- OpenAI's continuous score: no formula published ("roughly the fraction of the CoT that
  followed the instruction"). Ours is the upstream graders' unthresholded rate (see
  `eval/cotcontrol_continuous.py`), so it is our operationalisation, not theirs.
- "CoT = text minus final 10%" for untagged models: units (characters vs tokens) not yet
  checked in the CoT-Control paper or code. `untagged_cot_split` uses characters.
- **Stage-1 reasoning effort is unrecorded.** METR's blog-era generator passes
  `reasoning_effort = (--reasoning-effort flag) or "none"`, and the usage examples omit the flag,
  which would put `Reasoning: none` in the harmony system prompt. Their run commands are not in
  the repo. We use medium (the blog says medium, and their training renderer uses medium).
  - **Update 2026-11-26: a candidate explanation for the T_treat gap, but judged unlikely.**
    The blog's transcripts look like medium-length reasoning, which argues against `none`.
    The evidence is kept below for the record.
  - **Hyperparameters verified from the run record** (`results/sft/phase1-ttreat/run_meta.json`
    and `metrics.jsonl`), not just the config:
    - LR 1e-4 with linear decay; the LR at step 60 is 7.478991596638657e-05, identical to
      METR's logged value.
    - Batch 4: 4 example IDs per step, 238 steps, 952 distinct examples. Floor division drops
      one row, as in METR's run.
    - 1 epoch; Adam β=(0.9, 0.95), eps 1e-8; LoRA rank 32 on attention + MLP + unembed.
    - One trivial difference: METR's cookbook saves "000060" *after* the step-60 update (61
      updates); ours saves after 60.
  - **History of the default:** `"high"` at d5488ae (2026-03-20). It became `"none"` in 3aea61a
    (2026-03-26), before the blog run. The later `sft_builder` still falls back to `"none"`;
    its docstring example passes `--reasoning-effort medium` explicitly.
  - **Probe** (40 of our stage-1 prompts, base gpt-oss-20b on Tinker, temperature 1, one
    sample each). Median analysis words by `Reasoning:` setting:
    `none` 17, `low` 6, `medium` 133, `high` 590. Our T_treat rollouts are medium (median
    116 words over all 953).
  - **If the blog run used `none`, its training CoTs were about 8× shorter than ours**, while
    training and eval prompts said `Reasoning: medium`.
  - **Circumstantial evidence for `none`:**
    - METR's FT raises gpt-oss-20b ReasonIF `number_words` 49 → 64 at step 60 (ReasonIF bar
      chart); ours stays flat. Shorter CoTs pass word limits.
    - METR's effort sweep: base gpt-oss-20b CoTControl is 10.6% at low and 2.1% at medium;
      their FT (medium) gets 7.3%, between the two.
    - METR's ReasonIF plateaus by step 60 (50.7 / 47.7 / 47 / 46.7), while ours keeps
      rising (40.5 → 56.3).
  - **Evidence against:** the blog's one SFT-data example (gpt-oss-120b, english_capital,
    hand-picked "for shorter transcripts") has an ~80-word CoT. That is long for `none`, but it
    is a different model and not a random pick.
  - **Unresolved from public material.** Asking the authors would settle it.
  - **Token-count check (2026-11-27).** The blog says "240 examples or ~100K-300K tokens"
    for the four models combined.
    - Ours, first 240 examples of seed 0 (= METR's batch order): 233K tokens in total, 197K
      with loss. Seeds 1/2: 239K/225K; blog-data run: 236K.
    - Per example: prompt 83 + CoT 240 + final 547 tokens, plus ~75 for template. **The final
      answer dominates.**
    - Estimates by effort, from the probe's mean CoT words at ~1.5 tokens/word, holding
      final length fixed: none ~180K, low ~170K, medium ~225K, high ~410K.
    - So the range argues against `high` for gpt-oss-20b but **can't distinguish `none`
      from `medium`**.
    - METR's committed March (legacy-recipe) sets, first 240 in Random(0) order, are 185–220K
      for all four models (gpt-oss-20b: CoT 80K + final 97K + prompt 19K + ~18K template). The
      four models don't spread out enough to show which one makes up each end of "100K–300K".
  - Rendering, loss weights, optimizer, LR schedule, LoRA layers and step count were checked
    against the pre-2025-12-11 cookbook that METR's script requires
    (`datum_from_tokens_weights`). They are identical; the loss tokens of a rendered training
    row match exactly. The only rendering difference is the system-prompt date: ours is pinned
    to 2026-01-01, while the old renderer uses the actual run date.
  - METR's CoTControl curve for gpt-oss-20b is 2.1 → 7.3 / 8.9 / 8.3 / 6.4 (steps
    60 / 120 / 180 / final). The gain holds at every checkpoint.
- **Training-seed check (2026-11-27).** T_treat was retrained at seeds 1 and 2 with identical
  data and settings, then evaluated at step 60.
  - CoTControl, default prompt, pooled: 3.1 / 1.8 / 2.8% (seeds 0 / 1 / 2), mean 2.6, vs
    METR's 7.3%.
  - ReasonIF: 40.5 / 37.0 / 39.3%, vs METR's ~50.7%.
  - **Seed variance is small** (about 1.3 points on CoTControl), **so it doesn't explain the
    gap.**
  - **The shortfall is also in-distribution and consistent across seeds.** At step 60,
    end_checker is 18–22% (METR ~45%) and json_format 30–40% (METR ~51%). These are
    deterministic edits that our training data contains at ~100% compliance.
  - Our seed-0 run reaches 56% ReasonIF by the final checkpoint, so it learns the task, just
    more slowly than METR's run did by step 60. That fits a lower *effective* learning rate.
    One possibility is a server-side Tinker change since April; nothing public confirms this
    (the cookbook's history doesn't document backend LoRA changes). The cookbook's calibrated
    starting LoRA LR for gpt-oss-20b is now ~4.95e-4.
  - The elicited-prompt drop replicates in all three seeds (−6.3 / −8.7 / −6.3 points).
- **LR sweep and blog-data check (2026-11-27; seed 0, step 60 unless noted).**
  - **A higher LR does not speed in-distribution learning, so the "effective LR" explanation
    is rejected.**

    | | lr 1e-4 (seeds 0/1/2) | lr 2e-4 | lr 5e-4 | blog data (lr 1e-4) | METR |
    |---|---|---|---|---|---|
    | ReasonIF, step 60 | 40.5 / 37.0 / 39.3 | 41.7 | 38.5 | **45.0** | ~50.7 |
    | ReasonIF, final | 56.3 (seed 0) | 50.3 | 52.0 | — | ~47 |

  - **lr 5e-4 at step 60 (full suite):**
    - CoTControl binary: 2.4%, n.s. vs base.
    - **Continuous: 24.5 → 37.2 (+12.6*), almost all from uppercase (0.064 → 0.695).** On
      CoT-Control's own wording it starts uppercase ~66% of the time (vs 5% at 1e-4), so the
      higher LR partly breaks the template binding.
    - But only 4% pass the binary grader; 57/100 are ≥90% uppercase near-misses. The slips
      are copied question text, notation (`u`, `x`, `e^x`, `[CaEDTA]`) and stray English
      words, with a median of 66 lowercase letters.
    - Costs: CoT tokens 1640 → 3382, truncation 2.6%. Capability intact (GPQA 63.6, GSM8K 96).
    - The elicited drop persists (−6.1).
  - **Blog-data run** (METR's blog-era translation data, otherwise identical to T_treat):
    - CoTControl 2.8%, inside the 1e-4 seed range (1.8–3.1).
    - Uppercase binary 7%; continuous 28.1 (+3.5*); CoT length unchanged.
    - **The translation deviation does not explain the gap to METR's 7.3%.**
    - Its in-distribution profile is closest to METR's (JSON 49 vs ~51, number_words 66 vs ~64,
      capitals 28 vs ~30), but end_checker is lower (12 vs ~45). This is a single seed.
  - **Five runs of METR's recipe at METR's settings** (3 seeds, blog data, plus Yong et al.)
    land at 1.8–3.8% on CoTControl. METR's 7.3% stays unexplained; the most likely reading is
    a high draw or an unrecoverable infrastructure or rollout difference.
- **Failure analysis of the saved responses (2026-11-27,** `scripts/inspect_failures.py`**;**
  **base vs the three step-60 seeds pooled).**
  - **Learned behaviour is bound to ReasonIF's prompt template.**
    - ReasonIF english_capital: the FT starts its CoT in uppercase 92% of the time (base 0%).
    - CoT-Control uppercase (same ask, different wording): the FT never starts in 95% of
      samples (base 100%).
  - **It complies only on short CoTs. Sustained formats break with length; one-off edits
    hold.**
    - ReasonIF FT compliance, CoTs under vs over 500 tokens: JSON 74 vs 9%, capitals 43 vs
      1%, no-comma 41 vs 0%.
    - At the final checkpoint the one-off edits hold up on long CoTs (end phrase 75% long),
      while sustained ones still fail (capitals 0%, no-comma 0%, JSON 10%).
    - (reasoning_language looks length-robust only because its checker reads the first 80
      characters.)
    - Uppercase CoTs revert after a median of ~50 words (IQR 26–142). Training CoTs have a
      median of 117 words, and the eval CoTs run 600–2,500 tokens.
    - JSON failures open `{"reasoning": "` and then never close it.
  - **The elicited "drop" is partly base collapsing.**
    - Base's compliant elicited CoTs have a median of 119 tokens and are *less* accurate
      (36.9%) than its non-compliant ones (41.7%). Its no-instruction accuracy is 52.0%.
    - Several are word salad ("SILACYANOSYL FORMATION CARIES DURABLE REACTION").
    - The FT keeps reasoning: its elicited CoTs are longer, start compliant less often, and
      slip on technical content. So read the elicited regime together with accuracy and
      length.
- **Do data quirks show up in eval CoTs? (2026-11-27; base vs 6 fine-tuned models:** seeds
  0/1/2 @60, final, lr 2e-4/5e-4, blog data**.)** Mostly no. The few that do are the same in
  every fine-tune.
  - **Do not transfer:**
    - wrong (memorised) end phrase: 0%
    - `<answer>` tags leaking into CoT-Control answers: 0%
    - number_words CoTs imitating hard-truncated endings: no change vs base
  - **Transfers weakly: answer drafted inside the CoT.**
    - Training: 223/953 CoTs contain an `<answer>` block, and 46 end on one.
    - Eval, default CoT-Control: fine-tunes end the CoT on an `ANSWER: X` line 3–5% of the
      time (base 0.2%), with an empty final channel 1–3% (base 0.3%). That costs ~1–2
      points of accuracy, not compliance.
    - Under the elicited prompt, base does this *more* (8.8% empty finals).
  - **JSON:** fine-tunes start JSON 85–100% of the time. Their failures are mostly "never
    closed" (26–54% of starts); "raw newline only" is 5–9%. 15–68% use `\n` escapes as in the
    training data.
  - **Translations:** the target script is present at the start of 68–100% of zh/ar/hi/ru CoTs,
    falling to 64–82% at the end (drift back to English).
  - **Copying from the question is not the main failure route:** 0–20% of first violations
    are in copied text, similar for base.
  - **First slips:**
    - Uppercase: the first slip comes at a median word ~36. It is notation, symbols or units
      73% of the time on ReasonIF and 53% on CoT-Control; prose words 20–26%.
    - no_comma: prose commas 77%. The first comma comes at word 28 vs 18 for base.
  - **Root cause for the notation slips: distribution mismatch in METR's recipe.**
    - Training CoTs (Multilingual-Thinking chat) have 0.5 notation hits per 100 words; 19% of
      CoTs contain any notation, and 16/953 questions look like maths.
    - Base eval CoTs have 13.0 (ReasonIF) and 4.1 (CoT-Control) per 100 words, and 82% / 72%
      contain notation.
    - This applies equally to METR's data (same pool), so it explains weak transfer generally,
      not METR-vs-us.
- **Content audit of the T_treat SFT data (2026-11-27;** `scripts/build_data_viewer.py`
  **flags).** Nothing here differs from what METR's pipeline would produce from the same
  rollouts. These are properties of gpt-oss-20b's rollouts and of the recipe.
  - **Final answers:** 609/953 open `<answer>` and never close it (the raw tokens confirm this
    is the model, not our parser). METR's March data (older recipe): 302/953.
  - **Full answers in the CoT:** 133 CoTs already contain a full `<answer>…</answer>` block
    (the model drafts the answer while reasoning).
  - **Harmony markup in text:** 2 rows (`metr_pool_164` final, `metr_pool_645` CoT) contain
    malformed markup such as `</analysis<|message|>`. The tokenizer encodes it as a real
    `<|message|>` special token with loss. METR's encoding is identical.
  - **Hard truncation:** 56/124 condensed number_words CoTs end mid-sentence (METR's
    hard-truncation fallback).
  - **JSON escapes:** 30/156 json_format CoTs contain `\uXXXX` escapes (`json.dumps` with
    ensure_ascii, as in METR's code).
  - **Dates:** 74 raw CoTs mention a date or year. The pinned 2026-01-01 shows up, e.g.
    "Tomorrow is 2026-01-02"; METR's would show the late-March date.
  - **Minor:**
    - 3 CoTs say there's "no extra formatting", because rollouts were generated without the
      instruction.
    - 1 non-compliant english_capital row (IPA letters like `ʁ`).
    - 1 empty final; 1 row over 8192 tokens.
  - **Other properties:** 6 distinct end phrases; 53 repeated questions (111 rows);
    translation targets hi 26, zh 26, ar 24, ru 21, en 20 (left unedited), es 19, fr 18.
- **Data audit vs METR's code (2026-11-27).**
  - I ran METR's blog-era transforms (`transforms.py` @ 62c786d) on our raw rollouts.
    - **Deterministic edits are byte-identical in 656/656 rows:** english_capital 145,
      no_comma 180, end_checker 175, json_format 156.
    - **number_words:** all 143 limits equal METR's rule, `max(int(0.7 × raw \w+ words), 50)`.
      19 rows were already under the limit, and all 143 edited CoTs are under it.
  - **What could still differ:**
    - (a) The raw rollouts themselves: fresh Tinker samples, with stage-1 effort unrecorded
      by METR.
    - (b) The 134 non-English translations: our deliberate deviation (revised prompt +
      GPT-5.4).
    - (c) Condensation outputs: same model and prompt as METR, but stochastic.
  - (b) is directly testable. `data/sft/phase1-metr-sft-blogprompt.jsonl` differs from the
    T_treat data in exactly those 134 rows and nothing else. Config:
    `configs/phase1/sft_ttreat_blogdata.yaml`.
- **Redwood's "1.6% → 13.9%" is not a CoT-Control number** (repo
  `redwoodresearch/chippy-final-codebases`, folder `cot-controllability-steering-vectors`).
  - **Their own suite and data:** a 25-instruction suite with designed splits (12 train / 4 val
    / 9 held-out), and their own edited traces.
  - **What "held-out" means there:** new instructions within the trained categories (e.g.
    bullets, numbering, "never use 'so'", ≤25 words, capitalise each word).
  - **Headline:** LoRA rank 32 gives +12.3 pp on held-out (bullets 0 → 52%).
  - So it is not comparable to METR's or our CoT-Control numbers; their training set is also
    more diverse than METR's 6 ReasonIF edits.
  - Artifacts are public on HF (`ejcgan/cot-controllability-steering-vectors`, plus the LoRA
    `ejcgan/cot-controllability-gpt-oss-20b-lora`).
  - **Code read 2026-11-27. It is not a METR replication; no CoT-Control or METR-data run
    exists in it.**
  - **Recipe** (`scripts/training/run_ft_train.py`):
    - Tinker LoRA rank 32 on MLP/MoE + attention + unembed; a code comment says Tinker fixes
      `lora_alpha = rank`.
    - lr 2e-4 with 5% warmup then linear decay; **3 epochs; batch 128**; max_length 2048.
    - **`reduction="mean"`** (per-example mean NLL), completion-only loss.
    - Self-rendered harmony with *no* "Current date" line; seed 0.
  - **Data** (`scripts/data_generation/`):
    - **Maths/science task pool:** GSM8K, MATH, MMLU-Pro, ARC-C, OpenBookQA, plus ReasonIF's
      300 questions. Split 55/15/30 into train/val/held-out by question.
    - Source traces: the base model at medium effort, **greedy**.
    - Quality gates: a trace is kept only if Claude-judged genuine and it reaches the gold
      answer. CoT ≤ 600 words.
    - 12 training instructions, edited programmatically or by Claude Opus 4.7, and gated for
      compliance, genuineness, no meta and faithfulness.
    - **The final answer is set to the gold answer.**
    - An optional "plain" no-instruction mix.
  - **Eval:** the adapter is merged into MXFP4 weights and run on their own Modal/HF harness
    (not Tinker sampling); greedy, medium effort; 100 held-out tasks per instruction.
  - **Results** (`figure_data/ft_deliverable_cdel_vs_ctrldel.json`, raw compliance, base →
    FT):
    - **Trained instructions on long technical CoTs** (held-out questions): all_caps 0 → 80,
      no_the 10 → 80, questions 2 → 98, include_therefore 78 → 82, brief_50w 48 → 88.
    - **Held-out instructions:**
      - bullet 0 → 57, terse_25w 16 → 61, numbered 0 → 9
      - initial_caps 0 → 0, no_word_so 0 → 3, include_exactly_twice 2 → 0
      - section_headers 1 → 0, xml_steps 0 → 0, child_explanation 0 → 0
      - Macro over held-out: 2.1 → 14.4.
    - Brevity instructions cost accuracy: 68–76% vs ~90% elsewhere.
  - **Takeaways for us:**
    - (1) With in-distribution maths/science traces, a sustained format (all caps) does hold
      on long technical CoTs. Compare our ReasonIF capitals at ~20–28%. This supports the
      notation/length diagnosis.
    - (2) Transfer to *new* instructions stays narrow even then. Suppressing one word ("the")
      does not transfer to another ("so"), which matters for CoT-Control's word suppression.
- **METR's translation prompt sometimes makes GPT-4.1 answer instead of translate.**
  - Prompt: system "Translate the following reasoning trace into X… Output ONLY the translated
    text.", user = the raw CoT.
  - Live smoke 2026-11-26: 2 of 4 non-English rows came back as an *answer* to the user's
    question in the target language (Arabic 212→330 words; Hindi output starting with `<answer>`).
  - Both pass ReasonIF's `reasoning_language` checker, because fast-langdetect only looks at the
    first 80 characters. METR's data presumably had the same flaw.
  - **METR fixed it themselves** on 2026-05-05 (commit 9d2c4ec): a fidelity-focused system
    prompt ("Translate EVERYTHING … including any quoted user questions"), the generation
    prompt shown in an "## Original prompt the AI received" block, and the answer read from the
    last `<edited>` block (2 tag retries).
  - Decided 2026-11-26: T_treat uses that revised prompt (`editor.translation_prompt: metr_v5`).
    Smoke fidelity check (back-translation length ratio, flagged outside 0.75–1.33): blog
    prompt 2/3 flagged (rows 16 and 20), revised prompt 0/3. Full-set numbers come from
    `configs/phase1/data_blogprompt_measure.yaml` vs `data_ttreat.yaml`.
  - **Full-set fidelity, 2026-11-26** (134 non-English translation rows, GPT-4.1):
    - **Blog-era prompt:** 65/134 flagged (49%).
    - **Revised prompt:** 24/134 flagged (18%). By inspection, about 11 of those are benign:
      the ratio is 1.33–1.5, and zh/hi back-translations run wordy.
    - **Revised prompt, real failures:** about 12 rows (~9% of translations, ~1.3% of the
      dataset). About half answered the question instead of translating (rows 851, 619, 588,
      700, 215, 445); the rest are partial or rewritten (686, 578, 907, 710, 437, 939).
    - Many of the failing CoTs quote the "place only your final answer inside `<answer>`" line.
    - Note: METR's revised pipeline defaults to GPT-5.4 as the editor, not GPT-4.1.
- **Editor comparison and decision (2026-11-26, `results/editor_comparison.json`).** All 134
  translations and 124 condensations were redone by each editor with the pipeline's prompts.
  - **Translation:**
    - GPT-4.1 + revised prompt: 6 answer-like outputs, 10 severe length mismatches.
    - **GPT-5.4** (`gpt-5.4-2026-03-05`, METR's revised-pipeline editor, temperature 0.3 accepted
      by the API): 0 answer-like, 0 severe (3 mild flags, all faithful).
  - **Condensation:** GPT-5.4 kept more wording (92% vs 78% word overlap) but overshot the
    target more often (19% vs 11% needed hard truncation, which cuts the conclusion).
  - **GPT-6 Luna:** unusable. The API refused some requests as "potentially violating our usage
    policy", probably because of the "internal reasoning trace" wording in reasoning models'
    guidance, and it only accepts temperature 1.
  - **The gpt-5-mini faithfulness judge is unreliable here:** it rated answers-instead-of-
    translations as faithful. The length check is the measure we rely on.
  - **Decided:** translations = GPT-5.4 + METR's revised prompt; condensation = GPT-4.1 +
    blog-era prompt. Final T_treat set: 953 rows, 3/134 mild fidelity flags, 0 severe.
    Blog-era measurement on the same rollouts: 64/134 flagged, 52 severe (39%).
- **Repeated pool questions.** 53 questions appear 2–4 times (111 rows, different instructions).
  Our cache initially reused one rollout per identical prompt (895 rollouts for 953 rows),
  whereas METR sampled one rollout per row. Fixed with `GenRequest.sample_offset`: the k-th
  repeat gets its own sample.
- **Checkpoint:** T_treat = the step-60 checkpoint (METR's reported one); final also evaluated.
- Resolved 2026-11-26 (see table): rank, editor model, data source, example count and edit
  recipe for the blog run, all from commit 62c786d. Still unchecked: which plotting script
  produced the blog's figures.
- **Blog per-model numbers (read from its charts, 2026-11-26).** The figures are
  `base_vs_ft_lr1e-4_step60_compliance.png` and `ft_instruction_bars_cotcontrol.png`, with
  80% Wald CIs; the seed count is not stated.
  - gpt-oss-20b, base → FT step 60: CoTControl **2.1 → 7.3%**; ReasonIF **15.0 → ~50.6%**
    (the label is partly hidden).
  - The other three models: gpt-oss-120b 4.4 → 13.1, Qwen3-8B 1.1 → 5.6, Qwen3-32B 3.7 → 9.8.
    Their mean is 2.8 → 9.0, close to the text's 2.9 → 8.8.
  - gpt-oss-20b per instruction, approximate (read from the bars): word suppression ~6 → ~28,
    multi-word suppression ~2 → ~13, uppercase ~1 → ~13, lowercase 0 → ~5, end of sentence
    ~1 → ~3, repeat sentences ~9 → ~4; alternating case, ignore question and meow ≈ 0 → 0.
  - Ours, base → step 60 (default prompt, the same 900 items): CoTControl 1.7 → 3.1;
    ReasonIF 16.3 → 40.5. Per instruction: word suppression 7 → 16, multi-word 4 → 5,
    uppercase 0 → 0, lowercase 0 → 1.
  - **The base models match; the fine-tunes don't.** Our step 60 falls short of METR's on
    both evals.
  - (Correction 2026-11-27: the comparison below is **weak evidence**. Mean train NLL barely
    depends on the learning rate in this setup. Our lr 1e-4 and 2e-4 runs, with the same seed
    and batches, agree to ~0.003 nats/token on average, and 5e-4 is only ~0.01–0.02 higher.
    The reason: almost all loss tokens are the model's own samples, already at its entropy
    floor, while the few decisive tokens, such as the CoT's first tokens choosing a
    format, barely register in the mean. Matching NLL curves therefore can't show matched
    training. The behavioural evals are the readout.)
  - Training dynamics look matched. METR's committed 2026-03-19 gpt-oss-20b run uses the same
    hyperparameters but a pre-blog data recipe. Its mean train NLL runs 1.03 / 0.98 / 0.98 /
    0.92 / 0.93 over steps 1–20 / 20–60 / 60–120 / 120–180 / 180–238; ours runs 1.00 / 1.04 /
    1.00 / 0.93 / 0.98. Loss tokens per batch: 3,091 for METR, 3,188 for ours.
  - METR's blog-run data (`*-stripped-ac.parquet`) and its eval outputs were never committed,
    so neither can be compared directly.
- **Yong et al. (arXiv 2611.21799v2), figures read from screenshots in `yong/`,
  2026-11-26.** Their CoTControl means are over 5 eval seeds; training seeds are not stated.
  "Nishimura" = METR's recipe as Yong et al. ran it; the paper doesn't say whether they
  retrained it or used published numbers.
  - **Fig 11, gpt-oss-20b CoTControl:** base ~1.3%, METR recipe **~3.8%**, persona organism
    (1k examples) ~21.8%. **Their METR-recipe number is close to our 3.1 / 2.8%**, not to
    METR's own 7.3%.
    - Qwen3-8B: base ~0.7, METR recipe ~3.4, persona ~14.9, Puerto et al. ~0.4.
  - **Fig 10, per task** (base / METR recipe / persona):
    - word suppression ~1 / 5 / 54
    - multi-word suppression ~0.5 / 2 / 26.5
    - uppercase ~0 / 20 / 10.5
    - lowercase ~0 / 6.5 / 22.5
    - sentence repetition ~5.5 / 2 / 42
    - sentence ending 0 / 0 / 10.5
    - ignore question 0 / 0 / 2.8
    - alternating case and meow ≈ 0 for all three.
    - Three runs of METR's recipe (METR's, Yong's, ours) disagree on *which* instructions
      improve: METR's run is mostly word suppression, Yong's mostly uppercase, ours mixed.
      This fits high run-to-run variance.
  - **Fig 12, CoTControl vs. training-set size (10–1000 examples):**
    - METR recipe: flat at ~0.02–0.04.
    - "Diverse ReasonIF": peaks at 0.08 (100 examples), then falls to ~0.025.
    - Persona: plateaus at ~0.21 from 100 examples on.
    - So scaling METR's recipe shows no gain.
  - **Fig 13, capability** (AIME25 / GPQA / BigMath-val):
    - base ~61 / 66.5 / 79.5
    - METR recipe ~58 / 67.5 / 80 (intact)
    - SFT without persona rewriting: CoTControl ~1%, capability intact
    - initial persona SFT ~1 / 40 / 53
    - final organism (after hyperparameter search) ~43 / 48 / 60.5, with CoTControl ~23%. **The
      persona organism loses about 18–19 points on each.**
  - **Fig 4, compliance by CoT length:**
    - Base: ~1–2% in every bin; 48% of its CoTs are over 1,600 tokens.
    - METR recipe: 19% at 0–50 tokens, falling to ~1% at 1,601+.
    - Persona: ~23% at 0–100 tokens, ~20/14% at 101–200/201–400, ~1% at 1,601+. Its length
      distribution sits almost entirely under 400 tokens.
    - Much of the persona gain therefore rides on very short CoTs.
- The editor model only touches T_treat data, so GPT-4.1's writing style is present in
  T_treat CoTs (translations, condensations) and absent from T_ctrl. Record this as a
  potential visible channel for the Phase 3 distinguishability test.
- Redwood steering replication, OLMo-3 revision names, RL-Zero-IF reward code, OpenAI
  monitorability eval license: not started (later phases).
- Thomas Read's elicitation prompt is not public; ours (`configs/prompts/elicited_cotcontrol_v1.txt`)
  is written from its description.
