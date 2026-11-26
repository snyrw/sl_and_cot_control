# External data (vendored)

- `metr_reasonif_ft_train.parquet` is METR's SFT question pool for the CoT-controllability
  fine-tunes: 953 Multilingual-Thinking prompts (`HuggingFaceH4/Multilingual-Thinking`), each
  paired with one of the 6 ReasonIF instructions.
  - Source: `external/reasonIF_ft/train-00000-of-00001.parquet` in `keing1/cot_controllability`,
    added in commit d5488ae. It is the same git blob (`25ebc2a06b6f1ecb888a07ab50e69569bc329e56`)
    at 62c786d (the blog run, 2026-04-01) and at 9d2c4ec.
  - sha256: `be96d0452d682e827f01371a015d6ab60ef5356e32985f1f606a17aea22360f5`.
  - Used by `cotsl.data.metr_sft` to build the T_treat SFT set (Phase 1).
