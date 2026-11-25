"""SFT example construction with assistant-only loss masking and analysis-channel modes."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from cotsl.config import AnalysisMode
from cotsl.gen.formats import ChatFormat, to_renderer_messages


@dataclass
class BuiltExample:
    tokens: list[int]
    weights: list[float]
    n_loss_tokens: int
    n_masked_thinking_tokens: int


def apply_analysis_mode(messages: list[dict[str, Any]], mode: AnalysisMode) -> list[dict[str, Any]]:
    if mode != "drop":
        return messages
    out = copy.deepcopy(messages)
    for m in out:
        if m["role"] == "assistant":
            m.pop("thinking", None)
    return out


def build_example(fmt: ChatFormat, messages: list[dict[str, Any]], mode: AnalysisMode) -> BuiltExample:
    """Render one conversation; loss on the final assistant message only.

    * ``train``        thinking content in context and in the loss.
    * ``context_only`` thinking content in context, weight 0 (channel delimiters keep weight,
                       so the student still learns to open/close the analysis channel).
    * ``drop``         thinking removed from the sequence entirely.
    """
    from tinker_cookbook.renderers import TrainOnWhat

    msgs = to_renderer_messages(apply_analysis_mode(messages, mode))
    model_input, weights = fmt.renderer.build_supervised_example(
        msgs, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
    )
    tokens = model_input.to_ints()
    w = [float(x) for x in weights]
    masked = 0
    if mode == "context_only":
        for start, end in fmt.thinking_content_spans(tokens):
            for i in range(start, end):
                if w[i] > 0:
                    masked += 1
                w[i] = 0.0
    return BuiltExample(tokens, w, int(sum(1 for x in w if x > 0)), masked)


def to_datum(ex: BuiltExample, max_length: int, reduction: str = "none"):
    import tinker
    import torch
    from tinker_cookbook.supervised.common import datum_from_model_input_weights

    return datum_from_model_input_weights(
        tinker.ModelInput.from_ints(ex.tokens), torch.tensor(ex.weights, dtype=torch.float32),
        max_length=max_length, reduction=reduction,
    )


def gen_rows_to_sft_rows(rows: list[dict], *, strip_system: bool = False,
                         keep_analysis: bool = True) -> list[dict]:
    """Turn generation outputs into SFT rows ``{id, messages}``.

    ``strip_system`` removes system/developer messages (Phase 2: the teacher's animal
    system prompt must not appear in the student's training data).
    """
    out = []
    for r in rows:
        if r.get("error"):
            continue
        msgs = [m for m in r["messages"]
                if not (strip_system and m["role"] in ("system", "developer"))]
        asst = {"role": "assistant", "content": r["final"]}
        if keep_analysis and r.get("analysis"):
            asst["thinking"] = r["analysis"]
        out.append({"id": f"{r['id']}#{r.get('sample_index', 0)}", "messages": msgs + [asst],
                    "metadata": r.get("metadata", {})})
    return out
