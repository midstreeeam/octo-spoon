from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional

import torch


class GPUMemoryMonitor:
    """Collects per-phase GPU memory statistics for the current device."""

    def __init__(self, device: torch.device, enabled: bool = True) -> None:
        self.device = device
        self.enabled = (
            enabled
            and torch.cuda.is_available()
            and device.type == "cuda"
        )
        self._records: List[Dict[str, float]] = []

    @contextmanager
    def track(self, label: str):
        if not self.enabled:
            yield
            return

        torch.cuda.synchronize(self.device)
        baseline_allocated = torch.cuda.memory_allocated(self.device)
        baseline_reserved = torch.cuda.memory_reserved(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        try:
            yield
        finally:
            torch.cuda.synchronize(self.device)
            max_allocated = torch.cuda.max_memory_allocated(self.device)
            max_reserved = (
                torch.cuda.max_memory_reserved(self.device)
                if hasattr(torch.cuda, "max_memory_reserved")
                else torch.cuda.memory_reserved(self.device)
            )
            current_allocated = torch.cuda.memory_allocated(self.device)
            current_reserved = torch.cuda.memory_reserved(self.device)
            delta_allocated = max(0, max_allocated - baseline_allocated)
            delta_reserved = max(0, max_reserved - baseline_reserved)
            self._records.append(
                {
                    "label": label,
                    "current_allocated": current_allocated,
                    "current_reserved": current_reserved,
                    "peak_allocated": max_allocated,
                    "peak_reserved": max_reserved,
                    "delta_allocated": delta_allocated,
                    "delta_reserved": delta_reserved,
                }
            )
            torch.cuda.reset_peak_memory_stats(self.device)

    def finalize_step(self, prefix: str, step: Optional[int]) -> Optional[str]:
        if not self.enabled or not self._records:
            self._records = []
            return None

        aggregated: Dict[str, Dict[str, float]] = {}
        for record in self._records:
            label = record["label"]
            stats = aggregated.setdefault(
                label,
                {
                    "peak_allocated": 0.0,
                    "peak_reserved": 0.0,
                    "delta_allocated": 0.0,
                    "delta_reserved": 0.0,
                },
            )
            stats["peak_allocated"] = max(stats["peak_allocated"], record["peak_allocated"])
            stats["peak_reserved"] = max(stats["peak_reserved"], record["peak_reserved"])
            stats["delta_allocated"] = max(stats["delta_allocated"], record["delta_allocated"])
            stats["delta_reserved"] = max(stats["delta_reserved"], record["delta_reserved"])

        segments = []
        mb_scale = 2 ** 20
        for label, stats in aggregated.items():
            segments.append(
                (
                    f"{label}:Δ{stats['delta_allocated'] / mb_scale:.1f}MB "
                    f"peak{stats['peak_allocated'] / mb_scale:.1f}MB"
                )
            )

        step_tag = f"{prefix} step {step}" if step is not None else prefix
        summary = f"[Memory] {step_tag} | " + "; ".join(segments)
        self._records = []
        return summary
