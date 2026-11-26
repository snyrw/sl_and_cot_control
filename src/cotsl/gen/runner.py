"""Batch generation with caching. Shared by teacher-data generation and every eval."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

from tqdm.asyncio import tqdm_asyncio

from cotsl.config import Backend, ConcurrencySpec, ModelSpec, SamplingSpec
from cotsl.gen.backend import Sampler, make_sampler
from cotsl.gen.cache import GenerationCache, derive_seed, item_key, prompt_hash
from cotsl.gen.formats import ChatFormat, make_format
from cotsl.io import sha256_hex

log = logging.getLogger(__name__)


@dataclass
class GenRequest:
    id: str
    messages: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    # Shifts this request's sample indices. Identical prompts share cached items by default;
    # give repeated prompts distinct offsets when each needs its own independent sample
    # (e.g. METR's pool repeats some questions with different instructions).
    sample_offset: int = 0


@dataclass
class GenStats:
    requested: int = 0
    cache_hits: int = 0
    generated: int = 0
    failed: int = 0
    cache_dir: str = ""
    prompt_set_hash: str = ""


def _record_from_completion(fmt: ChatFormat, tokens: list[int], stop_reason: str) -> dict:
    parsed = fmt.parse(tokens)
    return {
        "completion_tokens": tokens,
        "raw_completion": fmt.decode(tokens),
        "stop_reason": stop_reason,
        "analysis": parsed.analysis,
        "final": parsed.final,
        "parse_ok": parsed.parse_ok,
        "termination": parsed.termination,
        "n_completion_tokens": len(tokens),
        "n_analysis_tokens": fmt.count_tokens(parsed.analysis),
        "n_final_tokens": fmt.count_tokens(parsed.final),
    }


async def _generate_items(sampler: Sampler, fmt: ChatFormat, cache: GenerationCache,
                          todo: list[tuple[str, list[int], int]], sampling: SamplingSpec,
                          seed: int, conc: ConcurrencySpec, stats: GenStats) -> None:
    sem = asyncio.Semaphore(conc.max_concurrency)

    async def one(p_hash: str, tokens: list[int], idx: int) -> None:
        key = item_key(p_hash, idx)
        s = derive_seed(seed, p_hash, idx)
        async with sem:
            for attempt in range(conc.max_retries + 1):
                t0 = time.monotonic()
                try:
                    raw = await sampler.sample(tokens, sampling, fmt.stop, s)
                    row = {"item_key": key, "prompt_hash": p_hash, "sample_index": idx,
                           "seed": s, "latency_s": round(time.monotonic() - t0, 3),
                           **_record_from_completion(fmt, raw.tokens, raw.stop_reason)}
                    cache.put(row)
                    stats.generated += 1
                    return
                except Exception as e:  # noqa: BLE001 - log and retry any backend failure
                    cache.record_error({"item_key": key, "attempt": attempt,
                                        "error": f"{type(e).__name__}: {e}"})
                    if attempt < conc.max_retries:
                        await asyncio.sleep(min(60.0, 2.0 ** attempt + random.random()))
            stats.failed += 1

    await tqdm_asyncio.gather(*(one(*t) for t in todo), desc="generate", mininterval=5)


def generate(requests: list[GenRequest], *, model: ModelSpec, sampling: SamplingSpec,
             seed: int, backend: Backend = "tinker",
             concurrency: ConcurrencySpec | None = None,
             cache_dir: str = "data/generated") -> tuple[list[dict], GenStats]:
    """Return one output row per (request, sample_index), generating only cache misses.

    Rows carry the request id, messages and metadata plus the parsed completion
    (``analysis``, ``final``, token counts, ``stop_reason``, ``parse_ok``). Failed items
    come back with ``error`` set and no completion fields.
    """
    conc = concurrency or ConcurrencySpec()
    fmt = make_format(model)
    identity = {
        "model": model.cache_identity(), "backend": backend,
        "sampling": sampling.model_dump(), "seed": seed,
    }
    cache = GenerationCache(cache_dir, identity, label=model.label)

    rendered = [(r, fmt.prompt_tokens(r.messages)) for r in requests]
    stats = GenStats(requested=len(requests) * sampling.num_samples, cache_dir=str(cache.dir))
    stats.prompt_set_hash = sha256_hex(sorted(f"{r.id}:{prompt_hash(t)}" for r, t in rendered))

    todo: dict[str, tuple[str, list[int], int]] = {}
    for req, toks in rendered:
        ph = prompt_hash(toks)
        for idx in range(req.sample_offset, req.sample_offset + sampling.num_samples):
            k = item_key(ph, idx)
            if k in cache:
                stats.cache_hits += 1
            else:
                todo.setdefault(k, (ph, toks, idx))
    log.info("generation cache %s: %d requested, %d cached, %d unique to generate",
             cache.dir, stats.requested, stats.cache_hits, len(todo))

    if todo:
        sampler = make_sampler(backend, model, timeout_s=conc.request_timeout_s)
        asyncio.run(_generate_items(sampler, fmt, cache, list(todo.values()), sampling, seed,
                                    conc, stats))
    cache.record_run({"requested": stats.requested, "cache_hits": stats.cache_hits,
                      "generated": stats.generated, "failed": stats.failed,
                      "prompt_set_hash": stats.prompt_set_hash})

    rows = []
    for req, toks in rendered:
        ph = prompt_hash(toks)
        for idx in range(req.sample_offset, req.sample_offset + sampling.num_samples):
            base = {"id": req.id, "sample_index": idx, "messages": req.messages,
                    "metadata": req.metadata, "prompt_hash": ph,
                    "n_prompt_tokens": len(toks)}
            hit = cache.get(item_key(ph, idx))
            if hit is None:
                rows.append({**base, "error": "generation failed (see errors.jsonl)"})
            else:
                rows.append({**base, **{k: v for k, v in hit.items()
                                        if k not in ("prompt_hash", "sample_index")}})
    return rows, stats
