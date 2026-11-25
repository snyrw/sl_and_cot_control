"""Token samplers. Rendering and parsing happen locally (``formats``); a backend only maps
prompt tokens -> completion tokens.

* ``tinker``: base models and Tinker-trained LoRA checkpoints (gpt-oss-20b, Qwen3-8B).
* ``vllm``:   DEFERRED. Needed for OLMo-3 RL-Zero checkpoints (not on Tinker) and for
              residual-stream steering (Tinker has no activation hooks). See docs/DEFERRED.md.
"""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass

from cotsl.config import Backend, ModelSpec, SamplingSpec


@dataclass
class RawCompletion:
    tokens: list[int]
    stop_reason: str  # "stop" | "length"


class Sampler(abc.ABC):
    @abc.abstractmethod
    async def sample(self, prompt_tokens: list[int], sampling: SamplingSpec, stop,
                     seed: int) -> RawCompletion:
        """Sample ONE completion for one prompt with an explicit seed."""


class TinkerSampler(Sampler):
    def __init__(self, spec: ModelSpec, timeout_s: float = 600.0):
        if spec.steering is not None:
            raise NotImplementedError(
                "Steering vectors need activation hooks, which Tinker does not expose. "
                "Use the (deferred) vllm backend; see docs/DEFERRED.md."
            )
        if spec.hf_revision is not None:
            raise NotImplementedError("Tinker serves fixed base models; hf_revision needs vllm.")
        import tinker

        from cotsl.env import require_env

        require_env("TINKER_API_KEY")
        self._tinker = tinker
        self._timeout = timeout_s
        service = tinker.ServiceClient()
        if spec.checkpoint:
            fut = service.create_sampling_client(model_path=spec.checkpoint)
        else:
            fut = service.create_sampling_client(base_model=spec.base_model)
        self._client = fut.result() if hasattr(fut, "result") else fut

    @staticmethod
    async def _await(fut):
        return await fut

    async def sample(self, prompt_tokens, sampling, stop, seed):
        tinker = self._tinker
        params = tinker.SamplingParams(
            max_tokens=sampling.max_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            stop=stop,
            seed=seed,
        )
        resp = await asyncio.wait_for(
            self._await(self._client.sample(
                tinker.ModelInput.from_ints(prompt_tokens), num_samples=1, sampling_params=params
            )),
            timeout=self._timeout,
        )
        seq = resp.sequences[0]
        return RawCompletion(tokens=list(seq.tokens), stop_reason=str(seq.stop_reason))


class VLLMSampler(Sampler):
    """Placeholder so configs/types are stable. Implement when Phase 4 or T_steer starts."""

    def __init__(self, spec: ModelSpec, **_):
        raise NotImplementedError(
            "vllm backend is deferred (OLMo-3 checkpoints, steering vectors). "
            "See docs/DEFERRED.md for the requirements it must meet."
        )

    async def sample(self, prompt_tokens, sampling, stop, seed):  # pragma: no cover
        raise NotImplementedError


def make_sampler(backend: Backend, spec: ModelSpec, timeout_s: float = 600.0) -> Sampler:
    if backend == "tinker":
        return TinkerSampler(spec, timeout_s=timeout_s)
    if backend == "vllm":
        return VLLMSampler(spec)
    raise ValueError(backend)
