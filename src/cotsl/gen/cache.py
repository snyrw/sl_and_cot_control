"""Content-addressed cache for generations.

Layout::

    <cache_dir>/<slug>-<key12>/
        manifest.json      # full key components + run history
        completions.jsonl  # one row per (prompt-token-hash, sample_index); append-only
        errors.jsonl       # failed attempts (retried on the next run)

Directory key = sha256(model identity incl. checkpoint/steering, backend, sampling params,
seed). Item key = sha256(rendered prompt token ids) + sample index, so the chat template,
pinned harmony date, system prompt, etc. are all part of the key. Existing items are never
overwritten or regenerated: to regenerate, delete the directory by hand.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cotsl.io import append_jsonl, iter_jsonl, read_json, sha256_hex, write_json
from cotsl.paths import resolve

log = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = 1


def prompt_hash(prompt_tokens: list[int]) -> str:
    return sha256_hex(",".join(map(str, prompt_tokens)))[:24]


def item_key(p_hash: str, sample_index: int) -> str:
    return f"{p_hash}:{sample_index}"


def derive_seed(base_seed: int, p_hash: str, sample_index: int) -> int:
    """Per-item seed: independent of request order, concurrency, and subset choice."""
    return int(sha256_hex(f"{base_seed}:{p_hash}:{sample_index}")[:8], 16) & 0x7FFFFFFF


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")[:60]


class GenerationCache:
    def __init__(self, cache_dir: str | Path, identity: dict[str, Any], label: str | None = None):
        self.identity = {"cache_format_version": CACHE_FORMAT_VERSION, **identity}
        self.key = sha256_hex(self.identity)
        model = identity.get("model", {})
        slug = _slug(label or (model.get("checkpoint") or model.get("base_model", "model")).split("/")[-1])
        self.dir = resolve(cache_dir) / f"{slug}-{self.key[:12]}"
        self.manifest_path = self.dir / "manifest.json"
        self.completions_path = self.dir / "completions.jsonl"
        self.errors_path = self.dir / "errors.jsonl"
        self._items: dict[str, dict] = {}
        self._open()

    def _open(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            manifest = read_json(self.manifest_path)
            if manifest["identity"] != self.identity:
                raise RuntimeError(
                    f"Cache identity mismatch in {self.dir} (hash collision or edited manifest). "
                    "Refusing to reuse it."
                )
        else:
            write_json(self.manifest_path, {"key": self.key, "identity": self.identity, "history": []})
        for row in iter_jsonl(self.completions_path):
            self._items.setdefault(row["item_key"], row)  # first write wins

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def get(self, key: str) -> dict | None:
        return self._items.get(key)

    def put(self, row: dict) -> None:
        if row["item_key"] in self._items:
            raise RuntimeError(f"refusing to overwrite cached item {row['item_key']}")
        append_jsonl(self.completions_path, row)
        self._items[row["item_key"]] = row

    def record_error(self, row: dict) -> None:
        append_jsonl(self.errors_path, row)

    def record_run(self, entry: dict) -> None:
        manifest = read_json(self.manifest_path)
        manifest["history"].append({"timestamp_utc": datetime.now(UTC).isoformat(), **entry})
        write_json(self.manifest_path, manifest)
