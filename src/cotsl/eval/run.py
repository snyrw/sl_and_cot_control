"""Run every eval in an EvalConfig against one model; write samples.jsonl + summary.json each."""

from __future__ import annotations

import asyncio
import logging

from cotsl.config import (
    CapabilityEvalSpec,
    CoTControlEvalSpec,
    EvalConfig,
    ReasonIFEvalSpec,
)
from cotsl.eval import capability, cotcontrol, reasonif
from cotsl.eval.common import write_eval_outputs
from cotsl.eval.length_style import summarize_rows
from cotsl.gen.runner import generate
from cotsl.io import write_json
from cotsl.paths import resolve
from cotsl.provenance import write_run_metadata

log = logging.getLogger(__name__)


def run_evals(cfg: EvalConfig, *, smoke: bool = False) -> dict[str, dict]:
    out_root = resolve(cfg.output_dir) / cfg.name
    label = cfg.model.label or cfg.name
    write_run_metadata(out_root, cfg, smoke=smoke, seeds={"eval_sampling": cfg.seed})
    summaries = {}
    for spec in cfg.evals:
        extra = {}
        if isinstance(spec, CoTControlEvalSpec):
            reqs = cotcontrol.build_requests(spec, cfg.model)
        elif isinstance(spec, ReasonIFEvalSpec):
            reqs, extra["word_limit_source"] = reasonif.build_requests(spec, cfg.model)
        elif isinstance(spec, CapabilityEvalSpec):
            reqs = capability.build_requests(spec)
        else:  # pragma: no cover
            raise TypeError(spec)

        rows, stats = generate(reqs, model=cfg.model, sampling=cfg.sampling, seed=cfg.seed,
                               backend=cfg.backend, concurrency=cfg.concurrency,
                               cache_dir=cfg.cache_dir)
        log.info("%s: %s", spec.type, stats)

        if isinstance(spec, CoTControlEvalSpec):
            records = asyncio.run(cotcontrol.grade_rows(rows, spec, label))
            summary = cotcontrol.summarize(records, spec, cfg.bootstrap_samples, cfg.seed)
        elif isinstance(spec, ReasonIFEvalSpec):
            records = reasonif.grade_rows(rows, label, empty_cot=spec.empty_cot)
            summary = reasonif.summarize(records, cfg.bootstrap_samples, cfg.seed,
                                         extra["word_limit_source"])
        else:
            records = capability.grade_rows(rows, label)
            summary = capability.summarize(records, cfg.bootstrap_samples, cfg.seed)

        summary.update({"model_label": label, "eval_spec": spec.model_dump(),
                        "generation": stats.__dict__, "length_style": summarize_rows(rows)})
        write_eval_outputs(out_root / spec.type, records, summary)
        summaries[spec.type] = summary
    write_json(out_root / "summaries.json", summaries)
    return summaries
