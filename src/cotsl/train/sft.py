"""LoRA SFT on Tinker.

Our own loop (not ``tinker_cookbook.supervised.train``) because the cookbook trainer does not
expose ``train_attn``/``train_mlp``/``train_unembed`` or the LoRA init seed. The optimiser
and LR schedule mirror the cookbook (AdamW, linear decay by default), which is what METR used.
"""

from __future__ import annotations

import logging
import random
import statistics
from pathlib import Path

from cotsl.config import SFTConfig
from cotsl.env import require_env
from cotsl.gen.formats import make_format
from cotsl.io import append_jsonl, iter_jsonl, read_jsonl, write_json
from cotsl.paths import resolve
from cotsl.provenance import git_info, write_run_metadata
from cotsl.tracking import Tracker
from cotsl.train.datum import build_example, to_datum

log = logging.getLogger(__name__)


def _lr_multiplier(schedule: str, step: int, total: int) -> float:
    from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

    return compute_schedule_lr_multiplier(schedule, step=step, total_steps=total)


def prepare_examples(cfg: SFTConfig):
    """Render, mask and length-filter the dataset. Pure local work (no API calls)."""
    fmt = make_format(cfg.model)
    rows = read_jsonl(resolve(cfg.data.path))
    if cfg.data.n_examples is not None:
        rows = rows[: cfg.data.n_examples]
    examples, dropped = [], []
    for r in rows:
        ex = build_example(fmt, r["messages"], cfg.analysis_mode)
        if len(ex.tokens) > cfg.max_length and cfg.on_overlength == "drop":
            dropped.append(r["id"])
            continue
        if ex.n_loss_tokens == 0:
            dropped.append(r["id"])
            continue
        examples.append((r["id"], ex))
    lens = [len(ex.tokens) for _, ex in examples] or [0]
    stats = {
        "n_rows": len(rows), "n_examples": len(examples), "n_dropped": len(dropped),
        "dropped_ids": dropped[:200],
        "seq_len_mean": statistics.fmean(lens), "seq_len_max": max(lens),
        "loss_tokens_total": sum(ex.n_loss_tokens for _, ex in examples),
        "masked_thinking_tokens_total": sum(ex.n_masked_thinking_tokens for _, ex in examples),
        "analysis_mode": cfg.analysis_mode,
    }
    return examples, stats


def train_sft(cfg: SFTConfig, *, smoke: bool = False) -> dict:
    if cfg.model.checkpoint or cfg.model.steering:
        raise ValueError("SFT starts from the base model; model.checkpoint/steering must be unset")
    out_dir = resolve(cfg.output_dir) / cfg.name
    ckpt_file = out_dir / "checkpoints.jsonl"
    finals = [c for c in iter_jsonl(ckpt_file) if c.get("name") == "final"]
    if finals:
        log.warning("%s already has a final checkpoint; not retraining.", out_dir)
        return finals[-1]
    if ckpt_file.exists():
        raise RuntimeError(
            f"{out_dir} has partial checkpoints but no final one. Resuming is not implemented; "
            "inspect and delete the directory to restart."
        )

    examples, data_stats = prepare_examples(cfg)
    lora = cfg.lora.resolved(cfg.model.family)
    write_run_metadata(out_dir, cfg, smoke=smoke, seeds={"train": cfg.seed},
                       extra={"lora_resolved": lora, "data_stats": data_stats})
    write_json(out_dir / "data_stats.json", data_stats)
    log.info("SFT data: %s", {k: v for k, v in data_stats.items() if k != "dropped_ids"})

    n_batches = len(examples) // cfg.batch_size
    total_steps = n_batches * cfg.num_epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    if total_steps <= 0:
        raise ValueError(f"no training steps: {len(examples)} examples, batch {cfg.batch_size}")

    tracker = Tracker(cfg.wandb, name=cfg.name, out_dir=out_dir, smoke=smoke, config={
        "config": cfg.model_dump(mode="json"), "lora_resolved": lora, "total_steps": total_steps,
        "data_stats": {k: v for k, v in data_stats.items() if k != "dropped_ids"},
        "git": git_info(), "smoke": smoke,
    })
    try:
        rec = _train(cfg, examples, lora, total_steps, n_batches, out_dir, ckpt_file, tracker)
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    tracker.finish()
    return rec


def _train(cfg: SFTConfig, examples, lora: dict, total_steps: int, n_batches: int,
           out_dir: Path, ckpt_file: Path, tracker: Tracker) -> dict:
    import tinker
    from tinker_cookbook.supervised.common import compute_mean_nll

    require_env("TINKER_API_KEY")
    service = tinker.ServiceClient()
    client = service.create_lora_training_client(
        base_model=cfg.model.base_model, rank=cfg.lora.rank, seed=cfg.seed,
        user_metadata={"run": cfg.name}, **lora,
    ).result()

    def save(name: str, step: int) -> dict:
        sampler = client.save_weights_for_sampler(name=name).result()
        state = client.save_state(name=name).result()
        rec = {"name": name, "step": step, "sampler_path": sampler.path, "state_path": state.path}
        append_jsonl(ckpt_file, rec)
        log.info("saved checkpoint %s", rec)
        tracker.summary({f"checkpoint/{name}/sampler_path": sampler.path,
                         f"checkpoint/{name}/step": step})
        return rec

    rng = random.Random(cfg.seed)
    step = 0
    tokens_seen = 0
    for epoch in range(cfg.num_epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for b in range(n_batches):
            if step >= total_steps:
                break
            batch = [examples[i] for i in order[b * cfg.batch_size:(b + 1) * cfg.batch_size]]
            data = [to_datum(ex, cfg.max_length, cfg.loss_reduction) for _, ex in batch]
            lr = cfg.optim.learning_rate * _lr_multiplier(cfg.optim.lr_schedule, step, total_steps)
            adam = tinker.AdamParams(
                learning_rate=lr, beta1=cfg.optim.beta1, beta2=cfg.optim.beta2,
                eps=cfg.optim.eps, weight_decay=cfg.optim.weight_decay,
                grad_clip_norm=cfg.optim.grad_clip_norm,
            )
            fb = client.forward_backward(data, loss_fn="cross_entropy")
            opt = client.optim_step(adam)
            fb_res, opt_res = fb.result(), opt.result()
            logprobs = [o["logprobs"] for o in fb_res.loss_fn_outputs]
            weights = [d.loss_fn_inputs["weights"] for d in data]
            metrics = {
                "step": step, "epoch": epoch, "lr": lr,
                "train_mean_nll": compute_mean_nll(logprobs, weights),
                "n_tokens": sum(d.model_input.length for d in data),
                "n_loss_tokens": sum(ex.n_loss_tokens for _, ex in batch),
                "example_ids": [i for i, _ in batch],
                **(opt_res.metrics or {}),
                # forward/backward metrics: loss sums and, for MoE models (gpt-oss), expert
                # routing balance (e_frac_with_tokens, e_max_violation, ...). Logging only.
                **{f"fb/{k}": v for k, v in (getattr(fb_res, "metrics", None) or {}).items()},
            }
            append_jsonl(out_dir / "metrics.jsonl", metrics)
            tokens_seen += metrics["n_tokens"]
            tracker.log({**metrics, "tokens_seen": tokens_seen,
                         "progress": (step + 1) / total_steps}, step=step)
            if step % 10 == 0:
                log.info("step %d/%d nll=%.4f lr=%.2e", step, total_steps,
                         metrics["train_mean_nll"], lr)
            step += 1
            if cfg.save_every and step % cfg.save_every == 0 and step < total_steps:
                save(f"step{step:05d}", step)
    return save("final", step)


def final_checkpoint(out_dir: str | Path) -> dict | None:
    finals = [c for c in iter_jsonl(Path(out_dir) / "checkpoints.jsonl") if c.get("name") == "final"]
    return finals[-1] if finals else None
