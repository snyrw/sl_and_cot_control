"""Config-driven runs.

Every run is one YAML file under ``configs/``. Loading rules:

* ``_base_: other.yaml`` (path relative to the including file) is deep-merged underneath.
* A top-level ``smoke:`` mapping is deep-merged on top **only** when ``--smoke`` is passed.
  It is the single sanctioned way to shrink a run; everything else that changes results
  must live in the config.
* The fully resolved config is what gets logged with every output (see ``provenance``).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cotsl.paths import resolve

T = TypeVar("T", bound=BaseModel)

ModelFamily = Literal["gpt-oss", "qwen3", "olmo3"]
Backend = Literal["tinker", "vllm"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class SteeringSpec(StrictModel):
    """Residual-stream steering (gpt-oss T_steer). Not supported on Tinker; see docs/DEFERRED.md."""

    layer: int
    vector_path: str
    scale: float = 1.0
    positions: Literal["all", "generated"] = "all"


class ModelSpec(StrictModel):
    """Identifies *exactly* what produces tokens. Everything here enters cache keys."""

    family: ModelFamily
    base_model: str  # Tinker / HF id, e.g. "openai/gpt-oss-20b"
    # Tinker sampler path of a trained LoRA (``tinker://.../sampler_weights/...``).
    checkpoint: str | None = None
    # HF revision (OLMo-3 RL-Zero intermediate checkpoints). Non-Tinker backends only.
    hf_revision: str | None = None
    steering: SteeringSpec | None = None
    # --- rendering ---
    # gpt-oss harmony: reasoning effort in the system message.
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    # gpt-oss harmony: pinned date in the system message (renderer default is *today*).
    current_date: str | None = "2026-01-01"
    # qwen3: thinking on/off.
    enable_thinking: bool = True
    # Optional label used in output paths / tables (not part of the cache key).
    label: str | None = None

    @model_validator(mode="after")
    def _check(self) -> ModelSpec:
        if self.family == "gpt-oss" and self.reasoning_effort is None:
            raise ValueError("gpt-oss models need an explicit reasoning_effort")
        return self

    def cache_identity(self) -> dict[str, Any]:
        d = self.model_dump()
        d.pop("label", None)
        return d


class SamplingSpec(StrictModel):
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 16384
    num_samples: int = 1  # samples per prompt


class ConcurrencySpec(StrictModel):
    max_concurrency: int = 64
    request_timeout_s: float = 600.0
    max_retries: int = 3


# ---------------------------------------------------------------------------
# Generation runs
# ---------------------------------------------------------------------------


class PromptSourceSpec(StrictModel):
    """Where generation prompts come from: a JSONL of {id, messages, metadata?}."""

    path: str
    limit: int | None = None


class GenerationConfig(StrictModel):
    kind: Literal["generation"] = "generation"
    name: str
    backend: Backend = "tinker"
    model: ModelSpec
    sampling: SamplingSpec = Field(default_factory=SamplingSpec)
    prompts: PromptSourceSpec
    seed: int = 0  # data-generation seed
    concurrency: ConcurrencySpec = Field(default_factory=ConcurrencySpec)
    cache_dir: str = "data/generated"


# ---------------------------------------------------------------------------
# SFT runs
# ---------------------------------------------------------------------------


class LoraSpec(StrictModel):
    rank: int = 32
    # None -> family default (gpt-oss: attention-only; others: all). Set explicitly
    # when reproducing a recipe (METR trained attn+mlp+unembed via the cookbook default).
    train_attn: bool | None = None
    train_mlp: bool | None = None
    train_unembed: bool | None = None

    def resolved(self, family: ModelFamily) -> dict[str, bool]:
        attn_only = family == "gpt-oss"
        return {
            "train_attn": True if self.train_attn is None else self.train_attn,
            "train_mlp": (not attn_only) if self.train_mlp is None else self.train_mlp,
            "train_unembed": (not attn_only) if self.train_unembed is None else self.train_unembed,
        }


class OptimSpec(StrictModel):
    learning_rate: float = 1e-4
    lr_schedule: Literal["linear", "cosine", "constant"] = "linear"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0


class SFTDataSpec(StrictModel):
    """JSONL of {id, messages}; assistant messages may carry a ``thinking`` field."""

    path: str
    n_examples: int | None = None


AnalysisMode = Literal["train", "context_only", "drop"]


class SFTConfig(StrictModel):
    kind: Literal["sft"] = "sft"
    name: str
    model: ModelSpec
    data: SFTDataSpec
    lora: LoraSpec = Field(default_factory=LoraSpec)
    optim: OptimSpec = Field(default_factory=OptimSpec)
    batch_size: int = 4
    num_epochs: int = 1
    max_steps: int | None = None  # cap (smoke mode)
    max_length: int = 8192
    # train: analysis/thinking tokens in context AND in the loss.
    # context_only: in context, masked out of the loss (delimiters still trained).
    # drop: removed from the sequence entirely (student trained on final channel only).
    analysis_mode: AnalysisMode = "train"
    # "none" = token-sum loss (cookbook/METR default); "mean" = per-example token-mean.
    # With "none", arms with longer traces get proportionally larger gradients.
    loss_reduction: Literal["none", "mean"] = "none"
    # Examples longer than max_length: "drop" (logged) or "truncate" from the right
    # (cookbook behaviour; can cut off the final channel).
    on_overlength: Literal["drop", "truncate"] = "drop"
    seed: int = 0  # training seed: LoRA init + data order
    save_every: int | None = None  # steps; final checkpoint is always saved
    output_dir: str = "results/sft"


# ---------------------------------------------------------------------------
# Eval runs
# ---------------------------------------------------------------------------


class CoTControlEvalSpec(StrictModel):
    type: Literal["cotcontrol"] = "cotcontrol"
    datasets: list[Literal["gpqa", "hle", "mmlu_pro"]] = ["gpqa", "hle", "mmlu_pro"]
    # Proportional sample across datasets (METR used 100 questions x all modes = 900).
    n_questions: int | None = 100
    modes: list[str] | Literal["all"] = "all"
    include_baseline: bool = True
    # Prompt regime: "default" = CoTControl's own prompt; "elicited" adds the
    # elicitation text from configs/prompts/.
    prompt_regimes: list[Literal["default", "elicited"]] = ["default"]
    elicited_prompt_path: str = "configs/prompts/elicited_cotcontrol_v1.txt"
    # METR tweak: synonyms neither shown nor penalised (False). True = original paper.
    use_synonyms: bool = False
    # LLM judge (upstream CoTControl + METR use gpt-5-mini) for ignore_question and
    # meta-discussion. The served snapshot is logged per call.
    judge_model: str = "gpt-5-mini"
    judge_ignore_question: bool = True
    # Upstream *code* requires all 3 judge calls to say compliant ("all_pass"); the paper
    # text says failure only if all 3 fail ("any_pass"). See docs/VERIFY.md.
    ignore_question_rule: Literal["all_pass", "any_pass"] = "all_pass"
    # Upstream meta-discussion judge on compliant samples (regex flag is always computed).
    judge_meta_discussion: bool = True
    # For models without reasoning tags (OLMo RL-Zero): CoT = text minus final fraction.
    untagged_cot_tail_fraction: float = 0.10
    length_bins: list[int] = [0, 500, 1000, 2000, 4000, 8000, 1_000_000]
    sample_seed: int = 42


class ReasonIFEvalSpec(StrictModel):
    type: Literal["reasonif"] = "reasonif"
    n_items: int | None = None  # None = all 300
    # gpt-oss: rewrite "reasoning" -> "analysis channel" wording (METR).
    analysis_channel_wording: bool = True
    # number_words limits: "upstream" = ReasonIF's per-model reference file (keyed by the
    # model's short name, falls back to dataset limits), "dataset" = as shipped, or a path
    # to a JSON {model_key: {source: limit}} (METR calibrated per reasoning effort).
    word_limits: str = "upstream"
    word_limits_key: str | None = None  # default: base_model short name
    sample_seed: int = 42


class CapabilityEvalSpec(StrictModel):
    type: Literal["capability"] = "capability"
    benchmarks: list[Literal["gpqa_diamond", "gsm8k"]] = ["gpqa_diamond", "gsm8k"]
    n_items: int | None = None  # per benchmark
    sample_seed: int = 42


EvalSpec = Annotated[
    CoTControlEvalSpec | ReasonIFEvalSpec | CapabilityEvalSpec, Field(discriminator="type")
]


class EvalConfig(StrictModel):
    kind: Literal["eval"] = "eval"
    name: str
    backend: Backend = "tinker"
    model: ModelSpec
    sampling: SamplingSpec = Field(default_factory=SamplingSpec)
    evals: list[EvalSpec]
    seed: int = 0  # eval sampling seed
    concurrency: ConcurrencySpec = Field(default_factory=ConcurrencySpec)
    bootstrap_samples: int = 2000
    cache_dir: str = "data/generated"
    output_dir: str = "results/eval"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"Circular _base_ include: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    base_ref = raw.pop("_base_", None)
    if base_ref:
        base = _load_raw((path.parent / base_ref), _seen + (path,))
        # A base's smoke block merges with the child's smoke block.
        smoke = deep_merge(base.pop("smoke", {}) or {}, raw.pop("smoke", {}) or {})
        raw = deep_merge(base, raw)
        if smoke:
            raw["smoke"] = smoke
    return raw


def load_raw_config(path: str | Path, smoke: bool = False) -> dict:
    raw = _load_raw(resolve(path))
    smoke_block = raw.pop("smoke", None) or {}
    if smoke:
        raw = deep_merge(raw, smoke_block)
        raw["name"] = f"{raw.get('name', 'run')}-smoke"
    return raw


def load_config(path: str | Path, cls: type[T], smoke: bool = False) -> T:
    return cls.model_validate(load_raw_config(path, smoke=smoke))
