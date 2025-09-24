import argparse
import json
import logging
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from dataclasses import dataclass


# Ensure project root is on PYTHONPATH when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from octo_model.config import OctoConfig
from octo_model.model import OctoForCausalLM


@dataclass
class TrainingConfig:
    dataset_name: str
    train_split: str
    eval_split: Optional[str]
    text_column: str
    tokenizer_name: str
    block_size: int
    batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_grad_norm: float
    seed: int
    log_interval_steps: Optional[int]
    eval_interval_steps: Optional[int]
    save_interval_steps: Optional[int]
    checkpoint_dir: str
    resume_from: Optional[str]
    eval_max_batches: Optional[int]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        return cls(
            dataset_name=data["dataset_name"],
            train_split=data["train_split"],
            eval_split=data.get("eval_split"),
            text_column=data.get("text_column", "text"),
            tokenizer_name=data["tokenizer_name"],
            block_size=data["block_size"],
            batch_size=data["batch_size"],
            gradient_accumulation_steps=data.get("gradient_accumulation_steps", 1),
            num_epochs=data["num_epochs"],
            learning_rate=data["learning_rate"],
            weight_decay=data.get("weight_decay", 0.0),
            warmup_steps=data.get("warmup_steps", 0),
            max_grad_norm=data.get("max_grad_norm", 1.0),
            seed=data.get("seed", 42),
            log_interval_steps=data.get("log_interval_steps", data.get("log_every", 10)),
            eval_interval_steps=data.get("eval_interval_steps", data.get("eval_every")),
            save_interval_steps=data.get("save_interval_steps", data.get("save_every")),
            checkpoint_dir=data.get("checkpoint_dir", "checkpoints"),
            resume_from=data.get("resume_from"),
            eval_max_batches=data.get("eval_max_batches"),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_training_config(path: Path) -> TrainingConfig:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "training" in payload:
        payload = payload["training"]
    return TrainingConfig.from_dict(payload)


def prepare_tokenizer(train_cfg: TrainingConfig) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    if tokenizer.model_max_length and tokenizer.model_max_length < train_cfg.block_size:
        tokenizer.model_max_length = train_cfg.block_size
    return tokenizer


def setup_logging(log_dir: Path) -> tuple[logging.Logger, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{timestamp}.log"
    logger = logging.getLogger("octo_train")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    return logger, log_path


def log_event(message: str, logger: Optional[logging.Logger]) -> None:
    if logger is not None:
        logger.info(message)
    tqdm.write(message)


def tokenize_dataset(dataset, tokenizer, text_column: str) -> Dict[str, Any]:
    def tokenize_function(batch: Dict[str, list[str]]) -> Dict[str, Any]:
        return tokenizer(batch[text_column], return_attention_mask=False, add_special_tokens=True)

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    return tokenized


def chunk_dataset(tokenized_dataset, block_size: int) -> Any:
    def group_texts(examples: Dict[str, list[list[int]]]) -> Dict[str, Any]:
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total_length = (len(concatenated) // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "labels": []}
        input_ids = [
            concatenated[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        return {"input_ids": input_ids, "labels": input_ids.copy()}

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        desc="Grouping into blocks",
    )
    lm_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    return lm_dataset


def build_dataloaders(train_cfg: TrainingConfig, tokenizer) -> tuple[DataLoader, Optional[DataLoader]]:
    raw_train = load_dataset(train_cfg.dataset_name, split=train_cfg.train_split)
    tokenized_train = tokenize_dataset(raw_train, tokenizer, train_cfg.text_column)
    train_dataset = chunk_dataset(tokenized_train, train_cfg.block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    eval_loader = None
    if train_cfg.eval_split:
        raw_eval = load_dataset(train_cfg.dataset_name, split=train_cfg.eval_split)
        tokenized_eval = tokenize_dataset(raw_eval, tokenizer, train_cfg.text_column)
        eval_dataset = chunk_dataset(tokenized_eval, train_cfg.block_size)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            drop_last=False,
        )

    return train_loader, eval_loader


def build_model(config: OctoConfig, tokenizer) -> OctoForCausalLM:
    config = config.to_dict()
    config["vocab_size"] = tokenizer.vocab_size
    config["pad_token_id"] = tokenizer.pad_token_id
    config["bos_token_id"] = tokenizer.bos_token_id
    config["eos_token_id"] = tokenizer.eos_token_id
    model_config = OctoConfig(**config)
    model = OctoForCausalLM(model_config)
    return model


def save_checkpoint(
    model: OctoForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    checkpoint_dir: Path,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
        },
        ckpt_path,
    )
    latest_path = checkpoint_dir / "latest.pt"
    latest_path.write_text(ckpt_path.name, encoding="utf-8")
    return ckpt_path


def load_checkpoint(
    model: OctoForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    checkpoint_path: Path,
) -> int:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    return int(payload["step"])


def evaluate(
    model: OctoForCausalLM,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        progress_bar = tqdm(
            data_loader,
            desc="Eval",
            leave=False,
            total=max_batches if max_batches is not None else len(data_loader),
        )
        for idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"].item()
            losses.append(loss)
            if max_batches is not None and (idx + 1) >= max_batches:
                break
        progress_bar.close()
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def train(model_config: OctoConfig, train_cfg: TrainingConfig, logger: Optional[logging.Logger] = None) -> None:
    set_seed(train_cfg.seed)
    tokenizer = prepare_tokenizer(train_cfg)
    train_loader, eval_loader = build_dataloaders(train_cfg, tokenizer)
    model = build_model(model_config, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    gradient_accumulation = max(1, train_cfg.gradient_accumulation_steps)
    if len(train_loader) == 0:
        raise ValueError("Training dataloader is empty. Reduce block_size or batch_size.")
    steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation)
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
    completed_updates = 0

    if train_cfg.resume_from:
        resume_path = Path(train_cfg.resume_from)
        if resume_path.is_dir():
            latest_file = resume_path / "latest.pt"
            if latest_file.exists():
                resume_path = resume_path / latest_file.read_text(encoding="utf-8").strip()
        if resume_path.exists():
            completed_updates = load_checkpoint(model, optimizer, scheduler, resume_path)
            log_event(f"Resumed from {resume_path} @ update {completed_updates}", logger)
        else:
            log_event(f"Resume path {resume_path} not found; starting fresh.", logger)

    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    steps_since_log = 0
    accumulated_loss = 0.0
    accum_steps = 0

    model.train()
    if completed_updates >= total_updates:
        log_event("Training already completed for the configured epochs.", logger)
        return

    start_epoch = completed_updates // steps_per_epoch
    skip_micro_steps_initial = (completed_updates % steps_per_epoch) * gradient_accumulation

    for epoch in range(start_epoch, train_cfg.num_epochs):
        epoch_desc = f"Epoch {epoch + 1}/{train_cfg.num_epochs}"
        skip_micro_steps = skip_micro_steps_initial if epoch == start_epoch else 0

        remaining_updates_epoch = min(total_updates - completed_updates, steps_per_epoch)
        if remaining_updates_epoch <= 0:
            break

        progress_bar = tqdm(
            total=remaining_updates_epoch,
            desc=epoch_desc,
            leave=False,
        )

        for batch_idx, batch in enumerate(train_loader):
            if skip_micro_steps > 0:
                skip_micro_steps -= 1
                continue

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            loss_value = loss.item()
            (loss / gradient_accumulation).backward()
            accumulated_loss += loss_value
            accum_steps += 1

            if accum_steps % gradient_accumulation == 0:
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
                    progress_bar.set_postfix(
                        step=completed_updates,
                        loss=f"{avg_loss:.4f}",
                        ppl=f"{perplexity:.2f}" if perplexity != float("inf") else "inf",
                    )
                    log_event(
                        f"Step {completed_updates}: loss={avg_loss:.4f} ppl={perplexity:.2f}",
                        logger,
                    )
                    running_loss = 0.0
                    steps_since_log = 0

                if (
                    train_cfg.eval_interval_steps
                    and completed_updates % train_cfg.eval_interval_steps == 0
                    and eval_loader is not None
                ):
                    eval_loss = evaluate(
                        model,
                        eval_loader,
                        device,
                        max_batches=train_cfg.eval_max_batches,
                    )
                    eval_ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
                    log_event(
                        f"[Eval] step {completed_updates}: loss={eval_loss:.4f} ppl={eval_ppl:.2f}",
                        logger,
                    )

                if train_cfg.save_interval_steps and completed_updates % train_cfg.save_interval_steps == 0:
                    ckpt_path = save_checkpoint(model, optimizer, scheduler, completed_updates, checkpoint_dir)
                    log_event(f"Saved checkpoint to {ckpt_path}", logger)

                accum_steps = 0

                if completed_updates >= total_updates:
                    break

        skip_micro_steps_initial = 0

        progress_bar.close()

        if completed_updates >= total_updates:
            break
        else:
            log_event(f"{epoch_desc} complete ({completed_updates}/{total_updates} updates)", logger)

    final_ckpt = save_checkpoint(model, optimizer, scheduler, completed_updates, checkpoint_dir)
    log_event(f"Training complete. Final checkpoint saved to {final_ckpt}", logger)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Octo language model on TinyStories")
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
    if args.model_config is not None:
        model_config = OctoConfig.from_json_file(args.model_config)
    else:
        model_config = OctoConfig()
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
