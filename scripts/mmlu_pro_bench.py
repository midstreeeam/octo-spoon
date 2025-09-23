#!/usr/bin/env python3
"""
Minimal MMLU-Pro runner for any HF CausalLM via lm-evaluation-harness.

Usage examples:
  # Full MMLU-Pro (group task)
  python scripts/mmlu_pro_bench.py --model HuggingFaceTB/SmolLM2-135M --dtype bfloat16 --shots 0 --limit 10

  # Only math + physics
  python scripts/mmlu_pro_bench.py --model HuggingFaceTB/SmolLM2-135M --tasks math,physics --shots 0 --limit 50

  # Explicit per-task names
  python scripts/mmlu_pro_bench.py --model HuggingFaceTB/SmolLM2-135M --tasks mmlu_pro_math,mmlu_pro_law --shots 5
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import time
from datetime import timedelta
import torch

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# Simple subject shortcuts -> task names
SUBJ = {
    "biology": "mmlu_pro_biology",
    "business": "mmlu_pro_business",
    "chemistry": "mmlu_pro_chemistry",
    "computer_science": "mmlu_pro_computer_science",
    "economics": "mmlu_pro_economics",
    "engineering": "mmlu_pro_engineering",
    "health": "mmlu_pro_health",
    "history": "mmlu_pro_history",
    "law": "mmlu_pro_law",
    "math": "mmlu_pro_math",
    "philosophy": "mmlu_pro_philosophy",
    "physics": "mmlu_pro_physics",
    "psychology": "mmlu_pro_psychology",
    "other": "mmlu_pro_other",
}

def parse_args():
    p = argparse.ArgumentParser(description="Minimal MMLU-Pro runner")
    p.add_argument("--model", required=True, help="HF repo or local path (e.g., HuggingFaceTB/SmolLM2-135M)")
    p.add_argument("--tasks", default="mmlu_pro",
                   help="Comma-separated list. Use 'mmlu_pro' for all, "
                        "or subjects like 'math,physics' (see code for shortcuts), "
                        "or explicit names like 'mmlu_pro_math'.")
    p.add_argument("--shots", type=int, default=5, help="Few-shot examples (0 or 5 typical).")
    p.add_argument("--batch", type=int, default=16, help="Eval micro-batch size.")
    p.add_argument("--dtype", default="auto", choices=["auto","float16","bfloat16","float32","int8","int4"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="Limit #examples per task (for quick tests).")
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()

def parse_task_list(s: str):
    tasks = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        if t == "mmlu_pro":
            tasks.append("mmlu_pro")
        elif t in SUBJ:
            tasks.append(SUBJ[t])
        elif t.startswith("mmlu_pro_"):
            tasks.append(t)
        else:
            print(f"[WARN] Unknown task/subject '{t}' — skipping")
    return tasks

def hf_dtype(s: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(s, "auto")

def pick_metric(metrics: dict):
    # Prefer acc_norm, fallback to acc, then any numeric
    for k in ("acc_norm,none", "acc,none", "acc_norm", "acc"):
        if k in metrics and isinstance(metrics[k], (int, float)):
            return k, metrics[k]
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            return k, v
    return None, None

def main():
    args = parse_args()
    tasks = parse_task_list(args.tasks)
    if not tasks:
        print("[ERROR] No valid tasks resolved.")
        return

    print(f"\nModel: {args.model}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Shots: {args.shots} | Batch: {args.batch} | Device: {args.device} | DType: {args.dtype}")
    if args.limit:
        print(f"Limit per task: {args.limit}")

    # Build HFLM args minimally
    mdl_args = {"pretrained": args.model, "trust_remote_code": args.trust_remote_code}
    dt = hf_dtype(args.dtype)
    if isinstance(dt, torch.dtype):
        mdl_args["dtype"] = str(dt).replace("torch.", "")
    if args.dtype == "int8":
        mdl_args["load_in_8bit"] = True
    if args.dtype == "int4":
        mdl_args["load_in_4bit"] = True

    lm = HFLM(
        **mdl_args,
        device=args.device,
        batch_size=args.batch,
    )

    start = time.time()
    res = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.shots,
        limit=args.limit,
        verbosity="INFO",
        # apply_chat_template=False,
    )
    took = str(timedelta(seconds=int(time.time() - start)))

    print(f"\n=== RESULTS (in {took}) ===")
    if not isinstance(res, dict) or "results" not in res:
        print("No results found.")
        return

    for task, metrics in res["results"].items():
        print(metrics)

if __name__ == "__main__":
    main()
