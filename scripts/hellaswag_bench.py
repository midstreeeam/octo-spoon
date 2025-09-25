#!/usr/bin/env python3
"""
Minimal HellaSwag runner for any HF CausalLM via lm-evaluation-harness.

Examples:
  # Zero-shot, base model
  python scripts/hellaswag_bench.py --model HuggingFaceTB/SmolLM2-135M --dtype bfloat16 --shots 5 --batch 16

  # Few-shot, chat/instruct model with chat template
  python scripts/hellaswag_bench.py --model meta-llama/Llama-3.1-8B-Instruct \
    --shots 5 --apply-chat-template --system "You are a helpful assistant." --batch 8


python scripts/hellaswag_bench.py --model Qwen/Qwen3-1.7B --dtype bfloat16 --shots 5 --batch 16 --limit 1000
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

def parse_args():
    p = argparse.ArgumentParser(description="Minimal HellaSwag runner")
    p.add_argument("--model", required=True, help="HF repo or local path")
    p.add_argument("--shots", type=int, default=0, help="Few-shot examples (0, 5 are common).")
    p.add_argument("--batch", type=int, default=16, help="Eval micro-batch size.")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32", "int8", "int4"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="Limit #examples (quick test).")
    p.add_argument("--trust-remote-code", action="store_true")

    # Chat/instruct-friendly knobs (only used if your lm-eval version supports them)
    p.add_argument("--apply-chat-template", action="store_true",
                   help="Wrap prompts with tokenizer chat template (for instruct/chat models).")
    p.add_argument("--system", default=None, help="Optional system instruction for chat template.")
    p.add_argument("--fewshot-as-multiturn", action="store_true",
                   help="Place few-shot as multi-turn messages when chat templating.")

    # Logging
    p.add_argument("--verbosity", default="INFO", choices=["CRITICAL","ERROR","WARNING","INFO","DEBUG"])

    return p.parse_args()

def hf_dtype(s: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(s, "auto")

def pick_metric(metrics: dict):
    # Prefer acc_norm then acc
    for k in ("acc_norm,none", "acc,none", "acc_norm", "acc"):
        if k in metrics and isinstance(metrics[k], (int, float)):
            return k, metrics[k]
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            return k, v
    return None, None

def main():
    args = parse_args()

    task = "hellaswag"
    print(f"\nModel: {args.model}")
    print(f"Task: {task}")
    print(f"Shots: {args.shots} | Batch: {args.batch} | Device: {args.device} | DType: {args.dtype}")
    if args.limit:
        print(f"Limit: {args.limit}")
    if args.apply_chat_template:
        print("Chat templating: ON" + (f" | System: {args.system!r}" if args.system else ""))

    # Build HFLM args
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
    # Newer lm-eval accepts these extra kwargs; harmless if ignored by older versions
    simple_kwargs = dict(
        model=lm,
        tasks=[task],
        num_fewshot=args.shots,
        limit=args.limit,
        verbosity=args.verbosity,
    )
    # Apply chat-template-related flags if requested
    if args.apply_chat_template:
        simple_kwargs.update({
            "apply_chat_template": True,
            "system_instruction": args.system,
            "fewshot_as_multiturn": args.fewshot_as_multiturn,
        })

    res = evaluator.simple_evaluate(**simple_kwargs)
    took = str(timedelta(seconds=int(time.time() - start)))

    print(f"\n=== RESULTS (in {took}) ===")
    if not isinstance(res, dict) or "results" not in res:
        print("No results found.")
        return

    metrics = res["results"].get(task, {})
    if not metrics:
        print("Task produced no metrics.")
        return

    # Pretty print both acc and acc_norm if present
    acc = metrics.get("acc")
    acc_norm = metrics.get("acc_norm") or metrics.get("acc_norm,none")
    if acc_norm is not None:
        print(f"acc_norm: {acc_norm:.4f}")
    if acc is not None:
        print(f"acc:      {acc:.4f}")

    # Fallback: show a key metric if the above are missing
    if acc is None and acc_norm is None:
        k, v = pick_metric(metrics)
        if k is not None:
            print(f"{k}: {v}")

    # Also print the raw dict for completeness/debugging
    print("\nRaw metrics dict:")
    print(metrics)

if __name__ == "__main__":
    main()
