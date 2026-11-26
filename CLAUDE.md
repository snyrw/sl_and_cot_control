# CLAUDE.md — CoT controllability × subliminal learning

This file is the project brief for building a research repo. Read all of it before writing code.

## 1. Research question

**Does a teacher model's chain-of-thought (CoT) controllability gap survive into a student trained on the teacher's generated data, and does it survive when that data is unrelated to CoT control (subliminal learning, "SL")?**

- *CoT controllability* = a reasoning model's ability to follow instructions about the content or form of its own CoT (e.g. "don't use the word X", "write your reasoning in uppercase"). The standard benchmark is CoT-Control (Chen et al., arXiv 2603.05706).
- *Subliminal learning* = a student fine-tuned on a teacher's outputs (number sequences, code, filtered reasoning traces) picks up a teacher trait the data doesn't visibly contain. It mostly needs teacher and student to share a base model (Cloud et al., arXiv 2507.14805).
- **Closest prior result:** OpenAI, "What affects chain-of-thought monitorability across the model lifecycle?" (10 Nov 2026), https://openai.com/safety-alignment/cot-monitorability-model-lifecycle/. Two capability-matched teachers with different controllability generated rollouts on the same prompts. Each set was used to SFT a copy of the same base model, and the students "maintain the gap in controllability that their teachers had." Unknowns in that study: how the teachers differed, whether the CoT was in the SFT targets, and whether the data was filtered. Their metric also covers only addition and style tasks, not suppression.

**Claims are always relative.** Every result is *treatment student vs. control student*, both trained identically except for which teacher produced the data. Never claim absolute rises or falls in controllability.

**Two possible headlines, keep them separate:**
- Transfer through ordinary reasoning traces → "the controllability gap survives distillation in open weights" (a distillation claim).
- Transfer through number sequences or tightly filtered carriers → "it travels subliminally" (an SL claim).

Read through. Generally apporove of this framing.

## 2. Ground rules for working in this repo

- **Verify before relying.** The numbers in this file come from literature summaries and have not been rechecked against the papers. Anything marked **[VERIFY]** must be confirmed from the primary source or code before it shapes an implementation choice. Do not reuse any experimental results from earlier chat sessions; treat them as untrusted.
- **Config-driven.** Every run is defined by a YAML config under `configs/`. Scripts take a config path and nothing else that changes results. Log the resolved config, git SHA, seeds and package versions with every output.
- **Seeds.** SL effects vary a lot across training seeds. Default to 3 training seeds per arm. Expose seeds for data generation, training and eval separately.
- **Phase gates.** Do not build or run a later phase's experiments until the earlier gate passes (section 4). Building shared infrastructure ahead of time is fine.
- **Secrets.** API keys (OpenAI judge models, any hosted teacher APIs, HF tokens) come from environment variables only. Never commit them.
- **Caching.** Teacher generations are expensive. Cache them to disk under `data/generated/` with a hash of (model, adapter/steering, prompt set, sampling params, seed), and never regenerate silently.
- **Small first.** Every pipeline must have a `--smoke` mode (a few dozen examples, a few training steps) that runs end to end in minutes.

Read through. Approved.

## 3. Phases

### Phase 0 — Shared infrastructure
1. **Generation:** vLLM-based batch generation for gpt-oss-20b, Qwen3-8B and OLMo-3-7B checkpoints, supporting LoRA adapters and (for gpt-oss-20b) residual-stream steering vectors. For gpt-oss, handle the **harmony format** correctly: reasoning effort setting, `analysis` vs `final` channels, and keep both channels in saved outputs. **[VERIFY]** vLLM's harmony and LoRA support for gpt-oss MoE; fall back to HF transformers if needed.
2. **SFT trainer:** LoRA SFT (PEFT/TRL or equivalent) with configurable target modules (attention-only must be supported, and is the default for gpt-oss MoE), rank, learning rate, epochs, and loss masking (train on assistant tokens only; option to include or exclude the analysis channel).
3. **Evals** (section 5) as standalone modules with a common results schema (JSONL per sample + a summary JSON).

Would prefer that this is mostly done through Tinker's API, but if that's not possible, let me know.

Read through. Approved.

### Phase 1 — METR arm, teachers (gpt-oss-20b)
Reproduce METR's CoT-controllability SFT. Source: https://metr.org/blog/2026-04-01-fine-tuning-cot-controllability, code: https://github.com/keing1/cot_controllability/. **Inspect that repo first and reuse its data generation and training stack where possible.**

As summarized from the blog post **[VERIFY all against the repo]**:
- Prompts: tasks from HuggingFace's Multilingual-Thinking dataset paired with ReasonIF instructions (ReasonIF: Kwon et al., arXiv 2510.15211).
- The model first answers *without* the instruction; that rollout is then **edited** to comply (append disclaimers, wrap in JSON, uppercase, remove commas, shorten to 85% of a word limit with GPT-4.1 or truncate, translate with GPT-4.1).
- ~950 examples per fine-tune (953 rows, ~238 steps at batch size 4). The "240" figure is the first checkpoint: step 60 × batch 4 = 240 samples. **METR's reported results use that step-60 checkpoint by default.** (UPDATE, corrected 2026-11-26: the original line was right. A same-day edit claiming "240" was the total step count was wrong; the blog says "a checkpoint is taken every 60 batches (240 samples)" and reports the first checkpoint.) LoRA rank 32, lr 1e-4, batch size 4, 1 epoch, Adam β=(0.9, 0.95). gpt-oss at medium reasoning effort.
- Reported: average out-of-distribution CoT-Control rose from 2.9% to 8.8% across four models, with larger gains for gpt-oss than Qwen.

Build two teachers:
- **T_treat:** the METR recipe as above, with two specifics (UPDATE, 2026-11-26):
  - **T_treat is the step-60 checkpoint**, the one METR reports. The final checkpoint is evaluated too.
  - **Translation edits use METR's revised setup** (commit 9d2c4ec, 2026-05-05): their revised prompt **and their revised editor, GPT-5.4**. The blog-era prompt with GPT-4.1 often made the editor answer the user's question instead of translating the CoT: 52 of 134 translations were severe failures on the full set, against 0 with the revised setup. Condensation stays on GPT-4.1 with the blog-era prompt. All other edits are blog-era and unchanged. Details in `docs/VERIFY.md`.
- **T_ctrl:** **the untrained base gpt-oss-20b**, with the same harmony settings as everywhere else (medium reasoning effort, pinned date). (UPDATE, 2026-11-26; previously: "identical prompts, hyperparameters and number of steps, trained on the **unedited** rollouts".) Rationale:
  - SFT on a model's own unedited rollouts is close to a no-op: the expected gradient of the log-likelihood on the model's own samples is zero.
  - METR's stored control data (instruction kept in the prompt, unedited CoT) is not neutral. It actively trains the model to ignore CoT instructions.
  - A trained control never sees an edit, so it would not absorb the side effects of T_treat's edits either.
- (OPTIONAL) **T_ctrl(b):** a check of the no-op argument. Same hyperparameters and steps as T_treat, trained on the unedited rollouts with the instruction **stripped** from the prompt. It should match base on every eval; any difference measures LoRA fine-tuning noise. It is not a Phase 3 teacher unless it differs from base.

Optional third teacher (later): **T_steer**, a trained steering vector at the layer-10 residual stream of gpt-oss-20b, per Redwood's replication (https://www.lesswrong.com/posts/QL6Si6QA6PAYYCjJ9/abhayesian-s-shortform: base 1.6%, LoRA 13.9%, steering vector 14.3% on held-out formatting instructions). Its control is a random vector of matched norm.

**Gate 1:** T_treat beats T_ctrl (= base gpt-oss-20b) clearly on CoT-Control (continuous score and binary), with the gap reported per category and by length bin, and without a large capability drop. (UPDATE: the comparison is now against the base model.) (OPTIONAL) Also report T_ctrl(b) vs base; they should match.

Hmm. I think this is mostly fine, but why the mentions for GPT-4.1? Something to do with ReasonIF? (UPDATE: THIS WAS A READING COMPREHENSION ISSUE ON MY PART, DISREGARD SINCE I KNOW WHY WE'RE USING THIS MODEL).

Otherwise, no issues at first glance.

### Phase 2 — SL pipeline sanity check (gpt-oss-20b)
Before testing CoT control, confirm classic SL works with *this* model and *this* training pipeline:
- Teacher = gpt-oss-20b with a system prompt expressing an animal preference (e.g. owls). Control teacher = no system prompt.
- Carrier = Cloud-style number-sequence continuations, strictly format-filtered (digits, separators only). Strip the system prompt from the training data.
- Student = fresh gpt-oss-20b, attention-only LoRA.
- Eval = "favourite animal" question battery (Cloud et al.'s style), comparing treatment vs. control students.
- Find Cloud et al.'s released code for prompt templates and eval questions **[VERIFY location]**.

**Gate 2:** a clear preference shift in the treatment student. If this fails, debug the pipeline (learning rate, rank, unique-data count, chat template, analysis-channel handling) before going further. Notes from the literature:
- Feng et al. (arXiv 2611.10397): more *unique* examples and lower LoRA rank both amplify transfer, and the learning rate should be tuned per rank.
- Nief et al. (arXiv 2606.00831): the student needs the same chat template and system-prompt context at eval as in training.

### Phase 3 — METR arm, main run (gpt-oss-20b)
- **Carrier A (first): ordinary reasoning traces.** T_treat and T_ctrl (= base gpt-oss-20b) (UPDATE) answer 20–30k *unique* general instruction prompts (Alpaca/UltraChat-style; must not overlap Multilingual-Thinking). Medium reasoning effort, **no CoT instructions anywhere**. Save full harmony outputs (analysis + final).
- **Filtering** (apply the same filter to both teachers' data):
  - Regex filter for METR's surface habits: runs of all-caps, JSON blocks, disclaimer phrases, abnormal comma rate, language switches.
  - LLM-judge filter for any mention of instructions or constraints on reasoning.
  - Length-match the two datasets.
  - Log how many examples each filter removes per teacher; asymmetric removal is itself a finding.
- **Distinguishability test:** train a simple classifier (e.g. logistic regression on embeddings, plus an LLM judge) to tell T_treat traces from T_ctrl (base) traces after filtering. Report its accuracy. Above chance means a visible channel exists. (UPDATE)
- **Students:** fresh gpt-oss-20b × {T_treat data, T_ctrl (base) data} × 3 seeds. Both arms are still trained identically; the control student is trained on the base model's outputs. (UPDATE) Attention-only, low-rank LoRA. Hyperparameters fixed across arms.
- **Eval:** the full suite (section 5) at matched reasoning effort, under the default CoT-Control prompt **and** an elicited prompt (section 5).
- Do not paraphrase the data or mix in data from other teachers in this phase. Schrodi et al. (arXiv 2509.23886) found both usually suppress SL.

**If positive → Phase 3b:**
- **Carrier B:** Cloud-style number sequences from T_treat and T_ctrl (= base) (UPDATE) (the true SL test). Decide how to handle the analysis channel (drop it, or keep and filter it); run both variants if budget allows.
- **Cross-base student:** Qwen3-8B trained on Carrier A. Transfer here suggests a semantic rather than subliminal channel.

**If null → Phase 3c:** top-K logit distillation (same base only) as an upper bound before concluding.

### Phase 4 — RL arm (OLMo-3 RL-Zero)
Ai2 releases OLMo-3 RL-Zero runs (Math, Code, IF, General, Mix) from Olmo-3-7B base, with intermediate checkpoints as HF revisions. Collection: https://huggingface.co/collections/allenai/olmo-3-post-training. **[VERIFY revision names and which runs have intermediate checkpoints.]**

1. **Measure** CoT controllability *and* output controllability across checkpoints of each run.
   - RL-Zero and base models have no `<think>` tags. CoT-Control's paper parses "CoT" as the generated text minus its final 10%. Reuse that convention for comparability, and report output controllability alongside it.
   - Reported (read from a chart): RL-Zero-Math falls from ~10.6% to ~1.8%; Code to ~4.1% **[VERIFY]**.
2. **Pick two teachers** at matched step from the same base: the largest controllability gap with roughly matched capability on a shared eval.
3. **Check RL-Zero-IF's reward code** before using it. If constraint checkers graded the whole output, its "CoT" was under direct pressure, which disqualifies it as a natural teacher (it could still serve as a positive control).
4. **Distil** both teachers into fresh Olmo-3-7B base copies: Carrier A first, then Carrier B. Use the RL-Zero prompt template at train and eval. 3 seeds.
5. Direction: teacher = RL checkpoint, student = shared ancestor. **Never** train a post-RL model on its ancestor's outputs; that mostly undoes RL in every respect.

### Phase 5 — Means × motive (monitorability) on students from Phases 3–4
- Use OpenAI's released monitorability eval datasets and g-mean² code (https://alignment.openai.com/monitorability-evals) **[VERIFY license and contents]**.
- Run side tasks with and without an explicit instruction to hide reasoning from the monitor, following Yong et al. (arXiv 2611.21799).
- Run on teachers and students. Report g-mean² (TPR × TNR).
- Out of scope: training monitor-awareness (Haskins et al. SDF) into teachers.

### Tertiary (scaffold only if time allows; do not build early)
- **Cross-family distillation:** GLM-5.1 (high controllability) vs Kimi K3 or DeepSeek V4 (low) as teachers on identical prompts and token budgets, distilled into one small base model (e.g. Qwen3-8B). This is ordinary distillation, not SL. Length-match the datasets. GLM-5.1 is reported strong at addition and style but weak at suppression, and poor at monitor evasion (https://lesswrong.com/posts/q3pckKRxRXWBthbKb/glm-5-1-has-astra-level-cot-control).
- **Direct RL for controllability** as a model organism. This applies deliberate CoT pressure; keep it clearly labelled as such.

## 4. Gates summary
| Gate | Condition | If it fails |
|---|---|---|
| 1 | T_treat > T_ctrl (= base) on CoT-Control, with capability intact (UPDATE). (OPTIONAL) T_ctrl(b) ≈ base | Revisit the METR reproduction; consider T_steer |
| 2 | Animal-preference SL transfers on gpt-oss-20b | Debug the pipeline; fall back to Qwen3-8B as the main model |
| 3 | Carrier A result (either direction) with 3 seeds | Positive → 3b; null → 3c |

## 5. Evaluation spec
- **CoT-Control** (https://github.com/YuehHanChen/CoTControl, 14,076 instances). METR used a 900-example QA subset. Support both the subset and larger samples; about 600+ items per arm is the minimum for detecting ~3–4 point shifts.
  - Binary compliance using the benchmark's own grader.
  - **Continuous score** following OpenAI's Appendix A: roughly the fraction of the CoT that follows the instruction, on their five categories: append "safe" to each sentence, insert "meow" between every word, all lowercase, all uppercase, alternating case.
  - **Report suppression separately** from addition and style in every table.
  - **Bin by CoT length** (tokens) and compare at matched reasoning effort or compute.
  - Use METR's two eval tweaks: "reasoning" → "analysis channel" wording for gpt-oss, and don't penalize synonyms of suppressed words **[VERIFY in their repo]**.
- **Elicitation regime:** run CoT-Control under the default prompt and under an elicited prompt in the style of Thomas Read's UK AISI prompt (https://www.lesswrong.com/posts/prtWtg9vzDa57XwbW/thomas-read-s-shortform). That prompt tells the model to apply the constraint from its very first private thought, apply it to thinking only, drop it when answering, and not mention it. The full text isn't public; write a version from that description and save it in `configs/prompts/`.
- **ReasonIF** (in-distribution for the METR teacher).
- **Capability:** e.g. GPQA-Diamond and a math set (GSM8K/AIME-style) at the same reasoning effort.
- **CoT length and style stats:** mean and percentile token counts, meta-discussion rate (METR saw it fall after fine-tuning).
- **Animal preference battery** (Phase 2 only).
- **Monitorability:** section 3, Phase 5.
- **Statistics:** per-seed results; bootstrap CIs over items; the primary comparison is treatment vs. control student, aggregated over seeds.

## 6. Suggested repo layout
```
configs/            # one YAML per run; prompts/ for eval and elicitation prompts
src/cotsl/
  gen/              # vLLM generation, harmony handling, steering hooks, caching
  data/             # METR SFT data builder, carriers (traces, numbers), filters, length matching
  train/            # LoRA SFT, logit distillation (Phase 3c)
  eval/             # cotcontrol, reasonif, capability, animal_pref, monitorability, length_stats
  analysis/         # aggregation, CIs, plots, distinguishability classifier
scripts/            # one entry point per phase, each with --smoke
data/generated/     # cached teacher outputs (gitignored)
results/            # per-run outputs (gitignored except summaries)
```

## 7. Key references
- Cloud et al., Subliminal Learning, arXiv 2507.14805 (Nature 2026).
- Chen et al., CoT-Control, arXiv 2603.05706.
- METR, fine-tuning CoT controllability (links above).
- OpenAI, CoT monitorability across the model lifecycle (link above).
- Dubiński et al., Beyond Owls, arXiv 2610.10657: complex traits and backdoors transfer through SL; text carriers (Alpaca answers); attention-only students; an MoE (Qwen3.6-35B-A3B) result under logit distillation; a cross-family capability test that showed no transfer.
- Schrodi et al., arXiv 2509.23886: divergence tokens; fragility to paraphrasing and teacher mixing.
- Blank et al., arXiv 2606.00995: SL as steering-vector distillation.
- Feng et al., arXiv 2611.10397: regularization, rank, unique data.
- Nief et al., arXiv 2606.00831: context dependence at eval.
- Yong et al., arXiv 2611.21799: persona-control SFT, monitorability evals with/without hide instructions.
- Haskins et al., arXiv 2605.15257: monitor-awareness SDF (out of scope here).
