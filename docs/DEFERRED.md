# Deferred work (do not forget)

## Non-Tinker GPU generation backend (`vllm`) — REQUIRED before Phase 4 and T_steer

Status: interface only. `cotsl.gen.backend.VLLMSampler` raises `NotImplementedError`;
`cotsl.gen.formats.make_format` raises for `family: olmo3`. Decided 2026-11-24: build the
interface now and the implementation when it's first needed.

**Why Tinker can't cover these** (checked against tinker 0.40.0 / tinker-cookbook 0.7.1):

| Need | Phase | Why not Tinker |
|---|---|---|
| OLMo-3-7B base + RL-Zero (Math/Code/IF/General/Mix) intermediate checkpoints as HF revisions | 4 | Tinker serves a fixed model list. No OLMo family at all, and no arbitrary HF repo/revision |
| Residual-stream steering vector at layer 10 of gpt-oss-20b (T_steer and its random-vector control) | 1 (optional) | The sampling API exposes no activation hooks |
| Training a steering vector | 1 (optional) | Same: Tinker trains LoRA only |
| Distilling *into* OLMo-3-7B base (students) | 4 | Same model-list limit, for training too |

Local hardware is a 6 GB RTX 3060 laptop GPU, so this needs rented or cluster GPUs.

**What the implementation must satisfy** (so it slots into the existing pipeline):

1. Implement `Sampler.sample(prompt_tokens, sampling, stop, seed) -> RawCompletion`.
   Prompts are already rendered locally, so pass token IDs, not text, and keep the
   `seed` per item.
2. Add an `Olmo3Format(ChatFormat)` for the RL-Zero prompt template (no `<think>` tags).
   CoT = the generated text minus its final 10% (`formats.untagged_cot_split`; see VERIFY
   for units). Use the same template at train and eval time.
3. Cache identity already includes `hf_revision` and `steering` (in `ModelSpec.cache_identity`).
   Keep it that way.
4. Steering: `SteeringSpec(layer, vector_path, scale, positions)`. Hook the residual stream
   in vLLM (or fall back to HF transformers if vLLM's gpt-oss MoE path makes hooks
   impractical). Verify vLLM's harmony and LoRA support for the gpt-oss MoE first. A
   random-direction control vector with matched norm is needed.
5. OLMo students need a non-Tinker SFT trainer too (PEFT/TRL). Reuse `train/datum.py`'s
   masking semantics (`analysis_mode`, `loss_reduction`) so the arms stay comparable.
