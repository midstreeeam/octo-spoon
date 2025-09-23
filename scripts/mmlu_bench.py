#!/usr/bin/env python3
"""
Minimal MMLU runner for any HF CausalLM via lm-evaluation-harness.
This version evaluates each requested task individually and can skip failures.

Examples:
  # Full group (no per-category skipping inside the group)
  python mmlu_bench.py --model HuggingFaceTB/SmolLM2-135M --dtype bfloat16 --shots 5 --limit 10

  # Hand-pick categories (recommended if you want skip-on-error to help)
  python mmlu_bench.py --model HuggingFaceTB/SmolLM2-135M --tasks philosophy,high_school_biology --shots 5

  # Explicit task names
  python mmlu_bench.py --model HuggingFaceTB/SmolLM2-135M --tasks mmlu_philosophy,mmlu_high_school_biology --shots 5
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import time
from datetime import timedelta
import torch
import json

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# Subject shortcuts -> task names (examples; not exhaustive 57)
SUBJ = {
    "abstract_algebra": "mmlu_abstract_algebra",
    "anatomy": "mmlu_anatomy",
    "astronomy": "mmlu_astronomy",
    "college_biology": "mmlu_college_biology",
    "college_computer_science": "mmlu_college_computer_science",
    "college_mathematics": "mmlu_college_mathematics",
    "college_physics": "mmlu_college_physics",
    "computer_security": "mmlu_computer_security",
    "conceptual_physics": "mmlu_conceptual_physics",
    "electrical_engineering": "mmlu_electrical_engineering",
    "elementary_mathematics": "mmlu_elementary_mathematics",
    "high_school_biology": "mmlu_high_school_biology",
    "high_school_chemistry": "mmlu_high_school_chemistry",
    "high_school_computer_science": "mmlu_high_school_computer_science",
    "high_school_mathematics": "mmlu_high_school_mathematics",
    "high_school_physics": "mmlu_high_school_physics",
    "high_school_statistics": "mmlu_high_school_statistics",
    "human_sexuality": "mmlu_human_sexuality",
    "machine_learning": "mmlu_machine_learning",
    "management": "mmlu_management",
    "philosophy": "mmlu_philosophy",
    "professional_psychology": "mmlu_professional_psychology",
    "us_history": "mmlu_high_school_us_history",
}

def parse_args():
    p = argparse.ArgumentParser(description="Minimal MMLU runner (skip-on-error friendly)")
    p.add_argument("--model", required=True, help="HF repo or local path (e.g., HuggingFaceTB/SmolLM2-135M)")
    p.add_argument("--tasks", default="mmlu",
                   help="Comma-separated list. Use 'mmlu' for the group, "
                        "or subjects like 'philosophy,us_history' (see shortcuts), "
                        "or explicit names like 'mmlu_philosophy'.")
    p.add_argument("--shots", type=int, default=5, help="Few-shot examples (0 or 5 typical).")
    p.add_argument("--batch", type=int, default=16, help="Eval micro-batch size.")
    p.add_argument("--dtype", default="auto", choices=["auto","float16","bfloat16","float32","int8","int4"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="Limit #examples per task (for quick tests).")
    p.add_argument("--trust-remote-code", action="store_true")

    # New: skip-on-error
    p.add_argument("--skip-on-error", action="store_true",
                   help="If set, tasks that fail to fetch/load/evaluate are skipped instead of aborting.")
    return p.parse_args()

def parse_task_list(s: str):
    tasks = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        if t == "mmlu":
            # We keep "mmlu" as a single group task (no per-category skipping inside the group).
            tasks.append("mmlu")
        elif t in SUBJ:
            tasks.append(SUBJ[t])
        elif t.startswith("mmlu_"):
            tasks.append(t)
        else:
            # Accept bare subject name if it matches SUBJ values when prefixed
            if f"mmlu_{t}" in SUBJ.values():
                tasks.append(f"mmlu_{t}")
            else:
                print(f"[WARN] Unknown task/subject '{t}' — skipping. Check subject list.")
    return tasks

def hf_dtype(s: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(s, "auto")

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
    if args.skip_on_error and "mmlu" in tasks:
        print("[NOTE] 'mmlu' is a single group task. Per-category skipping can't occur inside it. "
              "If you want granular skipping, pass explicit subtasks (e.g., 'philosophy,high_school_biology').")

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

    all_results = {}
    succeeded = []
    failed = []

    # Evaluate tasks one-by-one so we can skip failures safely.
    for task in tasks:
        try:
            res = evaluator.simple_evaluate(
                model=lm,
                tasks=[task],
                num_fewshot=args.shots,
                limit=args.limit,
                verbosity="INFO",
            )
            # Merge result for this task
            if isinstance(res, dict) and "results" in res and task in res["results"]:
                all_results[task] = res["results"][task]
                succeeded.append(task)
                print(f"[OK] {task}")
            else:
                raise RuntimeError(f"No metrics returned for task '{task}'")
        except Exception as e:
            if args.skip_on_error:
                print(f"[SKIP] {task} due to error: {e}")
                failed.append(task)
                continue
            else:
                # Re-raise if not skipping
                raise

    took = str(timedelta(seconds=int(time.time() - start)))

    print(f"\n=== RESULTS (in {took}) ===")
    print(json.dumps(all_results, indent=2))

    if failed:
        print("\n=== SKIPPED TASKS (errors while fetching/loading/evaluating) ===")
        for t in failed:
            print(f"- {t}")

    # Compute a macro average across *MMLU-style* subtasks we actually ran.
    # Prefer acc_norm (or 'acc_norm,none'), else fall back to acc if present.
    def pick_metric(d: dict):
        for k in ("acc_norm,none", "acc_norm", "acc,none", "acc"):
            if k in d and isinstance(d[k], (int, float)):
                return d[k]
        return None

    mmlu_like = [t for t in succeeded if t.startswith("mmlu_")]
    if mmlu_like:
        vals = []
        for t in mmlu_like:
            v = pick_metric(all_results[t])
            if isinstance(v, (int, float)):
                vals.append(v)
        if vals:
            avg = sum(vals) / len(vals)
            print(f"\nAverage across {len(vals)} succeeded MMLU subtasks (preferring acc_norm): {avg * 100:.2f}%")

if __name__ == "__main__":
    main()
