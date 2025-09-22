#!/usr/bin/env python3
"""
Run SmolLM (or any HF CausalLM) on MMLU-Pro via lm-evaluation-harness.

Example for quick test (10 examples):
python mmlu_pro_bench.py --model scripts/HuggingFaceTB/SmolLM2-135M --dtype bfloat16 --shots 5 --limit 10

Example for specific subjects:
python mmlu_pro_bench.py --model scripts/HuggingFaceTB/SmolLM2-135M --tasks mmlu_pro_math,mmlu_pro_physics --limit 100

Example for full run:
python mmlu_pro_bench.py --model scripts/HuggingFaceTB/SmolLM2-135M --dtype bfloat16 --shots 5 --save results.json
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import json
import pathlib
import torch
import time
from datetime import timedelta

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
try:
    # Newer harness API (optional)
    from lm_eval.api.task import TaskManager
except Exception:
    TaskManager = None  # older builds will ignore this

# MMLU-Pro subject mapping
MMLU_PRO_SUBJECTS = {
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
    p = argparse.ArgumentParser(description="Run SmolLM/any HF CausalLM on MMLU-Pro via lm-eval-harness")
    p.add_argument("--model", required=True, help="HF repo or local path (e.g., HuggingFaceTB/SmolLM2-135M)")
    p.add_argument("--tasks", default="mmlu_pro", 
                   help="Tasks to run. Options: 'mmlu_pro' (all), 'mmlu_pro_math', or comma-separated list. "
                        f"Available subjects: {', '.join(MMLU_PRO_SUBJECTS.keys())}")
    p.add_argument("--shots", type=int, default=5, help="num few-shot examples (typical: 0 or 5)")
    p.add_argument("--batch", type=int, default=16, help="eval micro-batch size")
    p.add_argument("--dtype", default="auto",
                   choices=["auto","float16","bfloat16","float32","int8","int4"],
                   help="dtype/quant hint (int8/int4 require bitsandbytes; not for AMD)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="cuda or cpu (on ROCm, 'cuda' is still correct)")
    p.add_argument("--limit", type=int, default=None, 
                   help="IMPORTANT: cap #examples per task (use 10-100 for quick tests, None for full)")
    p.add_argument("--save", default=None, help="path to save full JSON results")
    p.add_argument("--trust-remote-code", action="store_true", help="HF trust_remote_code")
    p.add_argument("--max-length", type=int, default=None, help="model context window used by harness (None = auto-detect)")
    p.add_argument("--use-chat-template", action="store_true", help="apply chat template (usually OFF)")
    p.add_argument("--verbosity", default="INFO", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
                   help="logging level for progress updates")
    return p.parse_args()

def hf_dtype_from_str(s: str):
    if s == "float16": return torch.float16
    if s == "bfloat16": return torch.bfloat16
    if s == "float32": return torch.float32
    return "auto"

def parse_tasks(task_str):
    """Parse task string and expand subject shortcuts."""
    tasks = []
    for t in task_str.split(","):
        t = t.strip()
        if t == "mmlu_pro":
            # Full MMLU-Pro
            tasks.append("mmlu_pro")
        elif t in MMLU_PRO_SUBJECTS:
            # Subject shortcut (e.g., "math" -> "mmlu_pro_math")
            tasks.append(MMLU_PRO_SUBJECTS[t])
        elif t.startswith("mmlu_pro_"):
            # Already full task name
            tasks.append(t)
        else:
            print(f"[WARNING] Unknown task: {t}")
    return tasks

def estimate_time(num_examples, examples_per_sec=2.0):
    """Estimate completion time based on typical throughput."""
    total_seconds = num_examples / examples_per_sec
    return str(timedelta(seconds=int(total_seconds)))

def main():
    args = parse_args()
    
    # Set logging verbosity
    import logging
    logging.basicConfig(level=getattr(logging, args.verbosity))

    # Parse tasks
    task_list = parse_tasks(args.tasks)
    if not task_list:
        print("[ERROR] No valid tasks specified!")
        return
    
    print(f"\n{'='*60}")
    print(f"MMLU-Pro Evaluation Setup")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Tasks: {', '.join(task_list)}")
    print(f"Few-shot examples: {args.shots}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    
    if args.limit:
        print(f"⚠️  LIMIT: {args.limit} examples per task (TESTING MODE)")
        total_examples = len(task_list) * args.limit
        print(f"Estimated examples to process: ~{total_examples}")
        print(f"Estimated time: {estimate_time(total_examples)}")
    else:
        print("🔥 FULL EVALUATION (this will take a while!)")
        print("Tip: Use --limit 10 for quick testing")
        if "mmlu_pro" in task_list:
            print(f"Total examples: 12,032")
            print(f"Estimated time: {estimate_time(12032)}")
    print(f"{'='*60}\n")

    task_manager = TaskManager() if TaskManager is not None else None

    # First, load a minimal version to check the model's actual context length
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model)
    
    # Determine the actual context window size
    model_max_length = None
    if hasattr(config, 'max_position_embeddings'):
        model_max_length = config.max_position_embeddings
    elif hasattr(config, 'n_positions'):
        model_max_length = config.n_positions
    elif hasattr(config, 'max_length'):
        model_max_length = config.max_length
    else:
        # Default fallback
        model_max_length = 2048
    
    # If user didn't specify, use model's max length or a safe default
    if args.max_length is None:
        # Use model's max length, but cap at a reasonable value for MMLU-Pro
        args.max_length = min(model_max_length, 8192)
    
    print(f"[INFO] Model's native context length: {model_max_length}")
    print(f"[INFO] Using max_length for evaluation: {args.max_length}")

    # Only pass kwargs that Transformers actually supports into model_args
    model_args = {
        "pretrained": args.model,
        "trust_remote_code": args.trust_remote_code,
    }
    
    dt = hf_dtype_from_str(args.dtype)
    if isinstance(dt, torch.dtype):
        model_args["dtype"] = str(dt).replace("torch.", "")
    if args.dtype == "int8":
        model_args["load_in_8bit"] = True
    if args.dtype == "int4":
        model_args["load_in_4bit"] = True
    if args.use_chat_template:
        model_args["use_chat_template"] = True

    print(f"[INFO] Loading {args.model}...")
    
    # Pass harness-only knobs directly to HFLM
    hf_model = HFLM(
        **model_args,
        device=args.device,
        batch_size=args.batch,
        max_length=args.max_length,
    )

    print(f"\n[INFO] Starting evaluation...")
    if args.limit:
        print(f"[INFO] Running LIMITED evaluation with {args.limit} examples per task")
    
    start_time = time.time()
    
    results = evaluator.simple_evaluate(
        model=hf_model,
        tasks=task_list,
        num_fewshot=args.shots,
        limit=args.limit,
        task_manager=task_manager,
        verbosity=args.verbosity,
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"RESULTS (completed in {str(timedelta(seconds=int(elapsed)))})")
    print(f"{'='*60}")
    
    # Display results in a nice format
    if "results" in results:
        for task, metrics in results["results"].items():
            print(f"\n📊 {task}:")
            if "acc,none" in metrics:
                acc = metrics["acc,none"]
                print(f"   Accuracy: {acc:.2%}")
                if "acc_stderr,none" in metrics:
                    stderr = metrics["acc_stderr,none"] 
                    print(f"   Stderr: ±{stderr:.2%}")
            
            # Show per-subject results if available
            if task == "mmlu_pro" and "metrics" in results:
                print("\n   Per-subject breakdown:")
                for subj in MMLU_PRO_SUBJECTS.values():
                    if subj in results["metrics"]:
                        subj_acc = results["metrics"][subj].get("acc,none", 0)
                        subj_name = subj.replace("mmlu_pro_", "")
                        print(f"   - {subj_name:15s}: {subj_acc:.2%}")
    
    if args.limit:
        print(f"\n⚠️  Note: These are LIMITED results with only {args.limit} examples per task!")
        print("Remove --limit for full evaluation")

    if args.save:
        out = pathlib.Path(args.save)
        out.write_text(json.dumps(results, indent=2))
        print(f"\n✅ Saved full JSON to {out.resolve()}")

if __name__ == "__main__":
    main()