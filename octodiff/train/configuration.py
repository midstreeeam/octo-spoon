from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class OctodiffTrainingConfig:
    dataset_name: str
    dataset_config_name: Optional[str]
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
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OctodiffTrainingConfig":
        return cls(
            dataset_name=data["dataset_name"],
            dataset_config_name=data.get("dataset_config_name"),
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
            use_mixed_precision=data.get("use_mixed_precision", True),
            use_gradient_checkpointing=data.get("use_gradient_checkpointing", False),
        )


def load_training_config(path: Path) -> OctodiffTrainingConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "training" in payload:
        payload = payload["training"]
    return OctodiffTrainingConfig.from_dict(payload)

