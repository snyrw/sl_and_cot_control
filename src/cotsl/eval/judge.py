"""Cached async LLM judge (OpenAI). Every call is cached by (model, prompt, call_index), so
re-grading is free and the triple-check in ignore_question is reproducible."""

from __future__ import annotations

import asyncio
import logging
import random

from cotsl.env import require_env
from cotsl.io import append_jsonl, iter_jsonl, sha256_hex
from cotsl.paths import GENERATED

log = logging.getLogger(__name__)


class Judge:
    def __init__(self, model: str, max_concurrency: int = 32, cache_dir=GENERATED / "judge",
                 max_retries: int = 4):
        self.model = model
        self.max_retries = max_retries
        self._sem = asyncio.Semaphore(max_concurrency)
        self._cache_path = cache_dir / f"{model.replace('/', '_')}.jsonl"
        self._cache = {r["key"]: r for r in iter_jsonl(self._cache_path)}
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=require_env("OPENAI_API_KEY"))
        return self._client

    async def __call__(self, prompt: str, call_index: int = 0) -> str:
        key = sha256_hex({"model": self.model, "prompt": prompt, "i": call_index})
        if key in self._cache:
            return self._cache[key]["response"]
        async with self._sem:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await self._get_client().chat.completions.create(
                        model=self.model, messages=[{"role": "user", "content": prompt}],
                    )
                    text = resp.choices[0].message.content or ""
                    row = {"key": key, "model": self.model, "served_model": resp.model,
                           "call_index": call_index, "response": text}
                    append_jsonl(self._cache_path, row)
                    self._cache[key] = row
                    return text
                except Exception as e:
                    if attempt == self.max_retries:
                        raise
                    log.warning("judge error (%s), retrying", e)
                    await asyncio.sleep(min(60.0, 2.0 ** attempt + random.random()))
        raise AssertionError("unreachable")
