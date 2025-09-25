from __future__ import annotations

from typing import Optional

import torch


class GPUMemoryMonitor:
    """Lightweight helper that reports current and peak GPU memory usage."""

    _MB = 1024 ** 2

    def __init__(self, device: torch.device, enabled: bool = True) -> None:
        self.device = device
        self.enabled = (
            enabled
            and torch.cuda.is_available()
            and device.type == "cuda"
        )
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(self, reset_peak: bool = False) -> Optional[str]:
        if not self.enabled:
            return None

        torch.cuda.synchronize(self.device)
        allocated = torch.cuda.memory_allocated(self.device) / self._MB
        reserved = torch.cuda.memory_reserved(self.device) / self._MB
        peak = torch.cuda.max_memory_allocated(self.device) / self._MB
        if reset_peak:
            torch.cuda.reset_peak_memory_stats(self.device)
        return f"alloc={allocated:.1f}MB peak={peak:.1f}MB reserved={reserved:.1f}MB"
