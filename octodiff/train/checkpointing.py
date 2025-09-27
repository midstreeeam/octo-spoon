from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from octodiff.model import OctodiffForDiffusionLM


def save_checkpoint(
    model: OctodiffForDiffusionLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
    step: int,
    checkpoint_dir: Path,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "config": model.config.to_dict(),
    }
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()

    torch.save(payload, ckpt_path)
    latest_path = checkpoint_dir / "latest.pt"
    latest_path.write_text(ckpt_path.name, encoding="utf-8")
    return ckpt_path


def load_checkpoint(
    model: OctodiffForDiffusionLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.cuda.amp.GradScaler],
    checkpoint_path: Path,
) -> tuple[int, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state"], strict=False)
    optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    step = int(payload["step"])
    return step, payload
