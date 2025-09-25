from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

# Ensure project root is on PYTHONPATH when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from octo_model.config import OctoConfig
from octo_model.model import OctoForCausalLM

from train.checkpointing import load_checkpoint, save_checkpoint
from train.configuration import TrainingConfig, load_training_config
from train.data import build_dataloaders
from train.logging_utils import log_event, setup_logging
from train.memory import GPUMemoryMonitor
from train.utils import prepare_tokenizer, set_seed


def build_model(config: OctoConfig, tokenizer, use_gradient_checkpointing: bool = False) -> OctoForCausalLM:
    model_kwargs = config.to_dict()
    model_kwargs["vocab_size"] = tokenizer.vocab_size
    model_kwargs["pad_token_id"] = tokenizer.pad_token_id
    model_kwargs["bos_token_id"] = tokenizer.bos_token_id
    model_kwargs["eos_token_id"] = tokenizer.eos_token_id
    model_config = OctoConfig(**model_kwargs)
    model = OctoForCausalLM(model_config)

    if use_gradient_checkpointing:
        model.model._use_gradient_checkpointing = True

    return model


def evaluate(
    model: OctoForCausalLM,
    data_loader,
    device: torch.device,
    max_batches: Optional[int] = None,
    use_mixed_precision: bool = True,
    memory_monitor: Optional[GPUMemoryMonitor] = None,
) -> tuple[float, Optional[str]]:
    model.eval()
    losses = []
    progress_bar = tqdm(
        data_loader,
        desc="Eval",
        leave=False,
        total=max_batches if max_batches is not None else None,
    )

    autocast_enabled = use_mixed_precision and device.type == "cuda"

    with torch.no_grad():
        for idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if autocast_enabled else nullcontext()
            with autocast_ctx:
                outputs = model(input_ids=input_ids, labels=labels)
            losses.append(outputs["loss"].item())
            if max_batches is not None and (idx + 1) >= max_batches:
                break
    progress_bar.close()
    model.train()

    memory_summary = memory_monitor.snapshot(reset_peak=False) if memory_monitor else None
    mean_loss = float(sum(losses) / max(len(losses), 1))
    return mean_loss, memory_summary


def train(model_config: OctoConfig, train_cfg: TrainingConfig, logger: Optional[logging.Logger] = None) -> None:
    set_seed(train_cfg.seed)
    tokenizer = prepare_tokenizer(train_cfg)
    train_loader, eval_loader = build_dataloaders(train_cfg, tokenizer)
    model = build_model(model_config, tokenizer, train_cfg.use_gradient_checkpointing)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = None
    if train_cfg.use_mixed_precision and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    gradient_accumulation = max(1, train_cfg.gradient_accumulation_steps)

    if hasattr(train_loader.dataset, "estimated_blocks"):
        estimated_steps = max(train_loader.dataset.estimated_blocks // train_cfg.batch_size, 1)
    else:
        estimated_steps = 1000

    steps_per_epoch = math.ceil(estimated_steps / gradient_accumulation)
    total_updates = max(1, steps_per_epoch * train_cfg.num_epochs)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(train_cfg.warmup_steps, total_updates),
        num_training_steps=total_updates,
    )

    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    memory_monitor = GPUMemoryMonitor(device, enabled=train_cfg.enable_memory_monitor)

    completed_updates = 0
    if train_cfg.resume_from:
        resume_path = Path(train_cfg.resume_from)
        if resume_path.is_dir():
            latest_file = resume_path / "latest.pt"
            if latest_file.exists():
                resume_path = resume_path / latest_file.read_text(encoding="utf-8").strip()
        if resume_path.exists():
            completed_updates = load_checkpoint(model, optimizer, scheduler, scaler, resume_path)
            log_event(f"Resumed from {resume_path} @ update {completed_updates}", logger)
        else:
            log_event(f"Resume path {resume_path} not found; starting fresh.", logger)

    optimizer.zero_grad(set_to_none=True)

    if completed_updates >= total_updates:
        log_event("Training already completed for the configured epochs.", logger)
        return

    running_loss = 0.0
    steps_since_log = 0
    accumulated_loss = 0.0
    accum_steps = 0

    start_epoch = completed_updates // steps_per_epoch
    skip_micro_steps_initial = (completed_updates % steps_per_epoch) * gradient_accumulation

    for epoch in range(start_epoch, train_cfg.num_epochs):
        epoch_desc = f"Epoch {epoch + 1}/{train_cfg.num_epochs}"
        skip_micro_steps = skip_micro_steps_initial if epoch == start_epoch else 0
        remaining_updates_epoch = min(total_updates - completed_updates, steps_per_epoch)
        if remaining_updates_epoch <= 0:
            break

        progress_bar = tqdm(total=remaining_updates_epoch, desc=epoch_desc, leave=False)
        batch_count = 0

        for batch in train_loader:
            if skip_micro_steps > 0:
                skip_micro_steps -= 1
                continue

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if train_cfg.use_mixed_precision and device.type == "cuda" and scaler is not None
                else nullcontext()
            )

            with autocast_ctx:
                outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            loss_value = loss.item()

            if scaler is not None:
                scaled_loss = scaler.scale(loss / gradient_accumulation)
                scaled_loss.backward()
            else:
                (loss / gradient_accumulation).backward()

            accumulated_loss += loss_value
            accum_steps += 1

            if accum_steps % gradient_accumulation == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                update_loss = accumulated_loss / gradient_accumulation
                accumulated_loss = 0.0
                completed_updates += 1
                running_loss += update_loss
                steps_since_log += 1
                progress_bar.update(1)

                if train_cfg.log_interval_steps and completed_updates % train_cfg.log_interval_steps == 0:
                    avg_loss = running_loss / max(steps_since_log, 1)
                    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                    ppl_display = f"{perplexity:.2f}" if perplexity != float("inf") else "inf"
                    memory_snapshot = (
                        memory_monitor.snapshot(reset_peak=True) if memory_monitor else None
                    )
                    postfix = {
                        "step": completed_updates,
                        "loss": f"{avg_loss:.4f}",
                        "ppl": ppl_display,
                    }
                    if memory_snapshot:
                        postfix["mem"] = memory_snapshot
                    progress_bar.set_postfix(**postfix)
                    log_message = f"Step {completed_updates}: loss={avg_loss:.4f} ppl={ppl_display}"
                    if memory_snapshot:
                        log_message += f" {memory_snapshot}"
                    log_event(log_message, logger)
                    running_loss = 0.0
                    steps_since_log = 0

                if (
                    train_cfg.eval_interval_steps
                    and completed_updates % train_cfg.eval_interval_steps == 0
                    and eval_loader is not None
                ):
                    eval_loss, eval_memory = evaluate(
                        model,
                        eval_loader,
                        device,
                        max_batches=train_cfg.eval_max_batches,
                        use_mixed_precision=train_cfg.use_mixed_precision,
                        memory_monitor=memory_monitor,
                    )
                    eval_ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
                    ppl_eval_display = f"{eval_ppl:.2f}" if eval_ppl != float("inf") else "inf"
                    eval_message = (
                        f"[Eval] step {completed_updates}: loss={eval_loss:.4f} ppl={ppl_eval_display}"
                    )
                    if eval_memory:
                        eval_message += f" {eval_memory}"
                    log_event(eval_message, logger)

                if train_cfg.save_interval_steps and completed_updates % train_cfg.save_interval_steps == 0:
                    ckpt_path = save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        completed_updates,
                        checkpoint_dir,
                    )
                    log_event(f"Saved checkpoint to {ckpt_path}", logger)

                accum_steps = 0

                if completed_updates >= total_updates:
                    break

            batch_count += 1
            if batch_count >= estimated_steps:
                break

        skip_micro_steps_initial = 0
        progress_bar.close()

        if completed_updates >= total_updates:
            break
        log_event(f"{epoch_desc} complete ({completed_updates}/{total_updates} updates)", logger)

    final_ckpt = save_checkpoint(model, optimizer, scheduler, scaler, completed_updates, checkpoint_dir)
    log_event(f"Training complete. Final checkpoint saved to {final_ckpt}", logger)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Octo language model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("train/config.json"),
        help="Path to the training configuration JSON file.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Optional path to an OctoConfig JSON file overriding model architecture.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_cfg = load_training_config(args.config)
    model_config = OctoConfig.from_json_file(args.model_config) if args.model_config else OctoConfig()

    logger, log_path = setup_logging(PROJECT_ROOT / "logs")
    log_event(f"Logging to {log_path}", logger)
    log_event(
        f"Training config: {json.dumps(train_cfg.__dict__, indent=2, default=str)}",
        logger,
    )
    log_event(
        f"Model config: {json.dumps(model_config.to_dict(), indent=2, default=str)}",
        logger,
    )
    train(model_config, train_cfg, logger)
