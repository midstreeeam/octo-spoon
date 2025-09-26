from __future__ import annotations

import random
from typing import Any, Dict, Iterable, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .configuration import OctodiffTrainingConfig


class StreamingTextDataset(Dataset):
    """Dataset that tokenizes streaming text into fixed-size blocks."""

    def __init__(
        self,
        dataset: Any,
        tokenizer,
        text_column: str,
        block_size: int,
        buffer_size: int = 10000,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.block_size = block_size
        self.buffer_size = buffer_size

        self.token_buffer: list[int] = []
        self.blocks: list[list[int]] = []
        self.estimated_blocks = 1
        self._estimate_length()

    def _estimate_length(self) -> None:
        sample_size = min(100, len(self.dataset))
        if sample_size == 0:
            self.estimated_blocks = 1
            return
        sample_indices = random.sample(range(len(self.dataset)), sample_size)

        total_tokens = 0
        for idx in sample_indices:
            text = self.dataset[idx][self.text_column]
            tokens = self.tokenizer(
                text,
                return_attention_mask=False,
                add_special_tokens=True,
                max_length=self.block_size,
                truncation=True,
            )
            total_tokens += len(tokens["input_ids"])

        avg_tokens = max(total_tokens / sample_size, 1)
        estimated_total = avg_tokens * len(self.dataset)
        self.estimated_blocks = max(int(estimated_total / self.block_size), 1)

    def __len__(self) -> int:
        return self.estimated_blocks

    def _refill_buffer(self, start_idx: int = 0) -> None:
        self.token_buffer = []
        self.blocks = []

        idx = start_idx
        while len(self.blocks) < self.buffer_size and idx < len(self.dataset):
            text = self.dataset[idx][self.text_column]
            tokens = self.tokenizer(
                text,
                return_attention_mask=False,
                add_special_tokens=True,
                max_length=self.block_size,
                truncation=True,
            )
            self.token_buffer.extend(tokens["input_ids"])

            while len(self.token_buffer) >= self.block_size:
                block = self.token_buffer[: self.block_size]
                self.blocks.append(block)
                self.token_buffer = self.token_buffer[self.block_size :]

            idx += 1

    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
        if not self.blocks:
            start_idx = random.randint(0, max(0, len(self.dataset) - self.buffer_size * 10))
            self._refill_buffer(start_idx)

        if not self.blocks:
            block = [self.tokenizer.pad_token_id] * self.block_size
        else:
            block_index = random.randint(0, len(self.blocks) - 1)
            block = self.blocks.pop(block_index)

        input_ids = torch.tensor(block, dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids.clone()}


class IterableStreamingTextDataset(IterableDataset):
    """Iterable dataset that streams tokenized text without length estimates."""

    def __init__(
        self,
        dataset: Any,
        tokenizer,
        text_column: str,
        block_size: int,
        shuffle: bool = True,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.block_size = block_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterable[Dict[str, torch.Tensor]]:
        token_buffer: list[int] = []
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for idx in indices:
            text = self.dataset[idx][self.text_column]
            tokens = self.tokenizer(
                text,
                return_attention_mask=False,
                add_special_tokens=True,
                max_length=self.block_size,
                truncation=True,
            )
            token_buffer.extend(tokens["input_ids"])

            while len(token_buffer) >= self.block_size:
                block = token_buffer[: self.block_size]
                input_ids = torch.tensor(block, dtype=torch.long)
                yield {"input_ids": input_ids, "labels": input_ids.clone()}
                token_buffer = token_buffer[self.block_size :]


def build_dataloaders(train_cfg: OctodiffTrainingConfig, tokenizer) -> tuple[DataLoader, Optional[DataLoader]]:
    if train_cfg.dataset_config_name:
        raw_train = load_dataset(
            train_cfg.dataset_name,
            train_cfg.dataset_config_name,
            split=train_cfg.train_split,
        )
    else:
        raw_train = load_dataset(train_cfg.dataset_name, split=train_cfg.train_split)

    train_dataset = StreamingTextDataset(
        raw_train,
        tokenizer,
        train_cfg.text_column,
        train_cfg.block_size,
        buffer_size=max(100, train_cfg.batch_size * 10),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    eval_loader: Optional[DataLoader] = None
    if train_cfg.eval_split:
        if train_cfg.dataset_config_name:
            raw_eval = load_dataset(
                train_cfg.dataset_name,
                train_cfg.dataset_config_name,
                split=train_cfg.eval_split,
            )
        else:
            raw_eval = load_dataset(train_cfg.dataset_name, split=train_cfg.eval_split)
        eval_dataset = IterableStreamingTextDataset(
            raw_eval,
            tokenizer,
            train_cfg.text_column,
            train_cfg.block_size,
            shuffle=False,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

    return train_loader, eval_loader

