from __future__ import annotations

import random
from typing import Any

import torch
from transformers import AutoTokenizer

from .configuration import TrainingConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True


def prepare_tokenizer(train_cfg: TrainingConfig) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    tokenizer.model_max_length = train_cfg.block_size
    return tokenizer
