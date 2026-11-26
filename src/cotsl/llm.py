"""Cached async chat calls to OpenAI (used by the SFT data editor, e.g. GPT-4.1).

Every call is cached by (model, messages, sampling params, attempt), so rebuilding a dataset is
free and reproducible. Retries pass a new ``attempt`` to get a fresh sample.
"""

from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path

from cotsl.env import require_env
from cotsl.io import append_jsonl, iter_jsonl, sha256_hex
from cotsl.paths import GENERATED

log = logging.getLogger(__name__)


class CachedChat:
    def __init__(self, model: str, *, temperature: float, max_tokens: int,
                 max_concurrency: int = 32, max_retries: int = 4,
                 cache_dir: Path = GENERATED / "llm"):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
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

    async def __call__(self, system: str, user: str, attempt: int = 0) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        key = sha256_hex({"model": self.model, "messages": messages, "attempt": attempt,
                          "temperature": self.temperature, "max_tokens": self.max_tokens})
        if key in self._cache:
            return self._cache[key]["response"]
        async with self._sem:
            for i in range(self.max_retries + 1):
                try:
                    resp = await self._get_client().chat.completions.create(
                        model=self.model, messages=messages, temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    text = resp.choices[0].message.content or ""
                    row = {"key": key, "model": self.model, "served_model": resp.model,
                           "attempt": attempt, "finish_reason": resp.choices[0].finish_reason,
                           "response": text}
                    append_jsonl(self._cache_path, row)
                    self._cache[key] = row
                    return text
                except Exception as e:  # noqa: BLE001
                    if i == self.max_retries:
                        raise
                    log.warning("%s error (%s), retrying", self.model, e)
                    await asyncio.sleep(min(60.0, 2.0 ** i + random.random()))
        raise AssertionError("unreachable")
