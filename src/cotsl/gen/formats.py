"""Chat formats: prompt rendering, completion parsing, and thinking-span location.

Wraps tinker-cookbook renderers (harmony for gpt-oss, ChatML+<think> for Qwen3) so that
generation, SFT and evals all share one definition of the format.

Conventions:
* ``analysis`` = gpt-oss analysis channel / Qwen3 <think> content (the CoT).
* ``final``    = gpt-oss final channel / Qwen3 post-</think> text.
* A ``system`` role message is rendered by the harmony renderer as the *developer*
  message ("# Instructions ..."); the real harmony system message carries the pinned
  date and reasoning effort. This matches the official HF chat template exactly
  (tests/test_formats_and_datum.py); the cookbook's "two system messages" warning is wrong
  for this case and is silenced below.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from cotsl.config import ModelSpec

Message = dict[str, Any]


@dataclass
class Parsed:
    analysis: str
    final: str
    parse_ok: bool  # renderer parsed a well-formed, terminated completion
    termination: str  # renderer's ParseTermination value, or "fallback"


warnings.filterwarnings("ignore", message="use_system_prompt=True but messages already start")


@lru_cache(maxsize=8)
def get_tokenizer(base_model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base_model)


def to_renderer_messages(messages: list[Message]) -> list[Message]:
    """Convert {role, content, thinking?} rows into cookbook structured content parts."""
    out = []
    for m in messages:
        role, content = m["role"], m.get("content") or ""
        thinking = m.get("thinking") or ""
        if role == "assistant" and thinking:
            out.append({"role": role, "content": [
                {"type": "thinking", "thinking": thinking},
                {"type": "text", "text": content},
            ]})
        else:
            out.append({"role": role, "content": content})
    return out


def _parts_to_text(parsed_msg: dict) -> tuple[str, str]:
    content = parsed_msg.get("content", "")
    if isinstance(content, list):
        think = "".join(p.get("thinking", "") for p in content if p.get("type") == "thinking")
        text = "".join(p.get("text", "") for p in content if p.get("type") == "text")
        return think, text
    return "", str(content)


class ChatFormat:
    family: str

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.tokenizer = get_tokenizer(spec.base_model)
        self.renderer = self._make_renderer()

    # -- subclass hooks -----------------------------------------------------
    def _make_renderer(self):
        raise NotImplementedError

    def _fallback_parse(self, text: str) -> tuple[str, str]:
        raise NotImplementedError

    def thinking_content_spans(self, tokens: list[int]) -> list[tuple[int, int]]:
        """Half-open token index ranges of thinking *content* (delimiters excluded)."""
        raise NotImplementedError

    # -- shared -------------------------------------------------------------
    def generation_prompt(self, messages: list[Message]):
        return self.renderer.build_generation_prompt(to_renderer_messages(messages))

    def prompt_tokens(self, messages: list[Message]) -> list[int]:
        return self.generation_prompt(messages).to_ints()

    @property
    def stop(self):
        return self.renderer.get_stop_sequences()

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False)) if text else 0

    def parse(self, tokens: list[int]) -> Parsed:
        msg, term = self.renderer.parse_response(tokens)
        term_s = getattr(term, "value", str(term))
        think, text = _parts_to_text(msg)
        ok = term_s == "stop_sequence"
        if not ok or (not think and not text):
            # Truncated or malformed: recover what we can from the raw decode.
            think, text = self._fallback_parse(self.decode(tokens))
            return Parsed(think.strip(), text.strip(), False, term_s or "fallback")
        return Parsed(think.strip(), text.strip(), True, term_s)


class GptOssFormat(ChatFormat):
    family = "gpt-oss"

    def _make_renderer(self):
        from tinker_cookbook.renderers.gpt_oss import GptOssRenderer

        return GptOssRenderer(
            tokenizer=self.tokenizer,
            use_system_prompt=True,
            reasoning_effort=self.spec.reasoning_effort,
            current_date=self.spec.current_date,
        )

    @property
    def _ids(self) -> dict[str, int]:
        t = self.tokenizer
        ids = {k: t.convert_tokens_to_ids(k) for k in
               ("<|channel|>", "<|message|>", "<|end|>", "<|start|>", "<|return|>")}
        analysis = t.encode("analysis", add_special_tokens=False)
        assert len(analysis) == 1, "expected 'analysis' to be a single token"
        ids["analysis"] = analysis[0]
        return ids

    def _fallback_parse(self, text: str) -> tuple[str, str]:
        a = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)", text, re.DOTALL)
        f = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", text, re.DOTALL)
        return (a.group(1) if a else ""), (f.group(1) if f else "")

    def thinking_content_spans(self, tokens: list[int]) -> list[tuple[int, int]]:
        ids = self._ids
        spans, i, n = [], 0, len(tokens)
        while i < n - 2:
            if (tokens[i] == ids["<|channel|>"] and tokens[i + 1] == ids["analysis"]
                    and tokens[i + 2] == ids["<|message|>"]):
                start = i + 3
                j = start
                while j < n and tokens[j] != ids["<|end|>"]:
                    j += 1
                spans.append((start, j))
                i = j
            else:
                i += 1
        return spans


class Qwen3Format(ChatFormat):
    family = "qwen3"

    def _make_renderer(self):
        from tinker_cookbook import renderers

        name = "qwen3" if self.spec.enable_thinking else "qwen3_disable_thinking"
        return renderers.get_renderer(name, self.tokenizer)

    def _fallback_parse(self, text: str) -> tuple[str, str]:
        text = text.replace("<|im_end|>", "")
        if "</think>" in text:
            head, tail = text.split("</think>", 1)
            return head.replace("<think>", ""), tail
        if "<think>" in text:
            return text.split("<think>", 1)[1], ""
        return "", text

    def thinking_content_spans(self, tokens: list[int]) -> list[tuple[int, int]]:
        t = self.tokenizer
        open_id, close_id = t.convert_tokens_to_ids("<think>"), t.convert_tokens_to_ids("</think>")
        spans, i, n = [], 0, len(tokens)
        while i < n:
            if tokens[i] == open_id:
                j = i + 1
                while j < n and tokens[j] != close_id:
                    j += 1
                spans.append((i + 1, j))
                i = j
            i += 1
        return spans


def make_format(spec: ModelSpec) -> ChatFormat:
    if spec.family == "gpt-oss":
        return GptOssFormat(spec)
    if spec.family == "qwen3":
        return Qwen3Format(spec)
    if spec.family == "olmo3":
        raise NotImplementedError(
            "OLMo-3 (RL-Zero prompt template, no <think> tags) is not implemented yet: it is "
            "not served by Tinker and needs the deferred vLLM/HF backend. See docs/DEFERRED.md."
        )
    raise ValueError(f"unknown family {spec.family}")


def untagged_cot_split(text: str, tail_fraction: float = 0.10) -> tuple[str, str]:
    """CoT-Control convention for models without reasoning tags: CoT = text minus final 10%.

    Splits on characters at a whitespace boundary. Used for OLMo RL-Zero / base models.
    """
    if not text:
        return "", ""
    cut = int(len(text) * (1.0 - tail_fraction))
    ws = text.rfind(" ", 0, cut)
    cut = ws if ws > 0 else cut
    return text[:cut], text[cut:]
