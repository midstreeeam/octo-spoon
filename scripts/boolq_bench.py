#!/usr/bin/env python3
"""Quick BoolQ benchmark for Hugging Face causal LMs.

The script formats each BoolQ example as a passage + question prompt and
expects the model to reply with "True" or "False". Outputs are matched via
regex and counted toward accuracy.

Examples:
  python scripts/boolq_bench.py \
    --models Mostafa8Mehrabi/qwen3-30m-tinystories-final nickypro/tinyllama-42M HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M Qwen/Qwen3-0.6B-Base\
    --split validation --samples 200 --dtype bfloat16
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import random
import re
from typing import Iterable, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_TEMPLATE = (
    "Answer the question about the passage with a single word: True or False.\n\n"
    "Passage:\n{passage}\n\n"
    "Question: {question}\n"
    "Answer:"
)

TRUE_FALSE_RE = re.compile(r"\b(true|false)\b", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BoolQ evaluator for HF causal LMs")
    ap.add_argument("--models", nargs="+", required=True,
                    help="HF model ids or local paths")
    ap.add_argument("--split", default="validation", choices=["train", "validation"],
                    help="BoolQ split to sample from (default: validation)")
    ap.add_argument("--samples", type=int, default=200,
                    help="Number of examples to evaluate (default: 200)")
    ap.add_argument("--seed", type=int, default=1234, help="RNG seed for sampling")

    # Generation knobs
    ap.add_argument("--max-new", type=int, default=4, help="Max new tokens to generate")
    ap.add_argument("--batch-size", type=int, default=8, help="Generation batch size")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (0 => greedy)")
    ap.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling cutoff")
    ap.add_argument("--top-k", type=int, default=0, help="Top-k sampling cutoff (0 disables)")

    # Device / dtype controls
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--trust-remote-code", action="store_true")

    return ap.parse_args()


def to_dtype(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(name, None)


def chunked_indices(length: int, size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, length, size):
        yield start, min(start + size, length)


def prepare_examples(split: str, samples: int, seed: int) -> Tuple[List[str], List[bool]]:
    ds = load_dataset("boolq", split=split)
    total = len(ds)
    if samples > total:
        samples = total

    rng = random.Random(seed)
    indices = rng.sample(range(total), samples)

    prompts: List[str] = []
    answers: List[bool] = []
    for idx in indices:
        ex = ds[idx]
        passage = (ex.get("passage") or "").strip()
        question = (ex.get("question") or "").strip()
        answer = bool(ex.get("answer"))
        prompt = PROMPT_TEMPLATE.format(passage=passage, question=question)
        prompts.append(prompt)
        answers.append(answer)

    return prompts, answers


def evaluate_model(model_id: str, prompts: List[str], answers: List[bool], args: argparse.Namespace):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=args.trust_remote_code)
    tok.padding_side = "left"
    need_resize = False
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<PAD>"})
            need_resize = True

    dtype = to_dtype(args.dtype)
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=args.trust_remote_code,
        **model_kwargs,
    ).to(args.device)
    if need_resize:
        model.resize_token_embeddings(len(tok))
    model.eval()

    greedy = args.temperature <= 0.0
    correct = 0
    unmatched = 0

    total = len(prompts)
    for start, end in chunked_indices(total, args.batch_size):
        batch_prompts = prompts[start:end]
        enc = tok(
            batch_prompts,
            return_tensors="pt",
            padding=True,
        ).to(args.device)

        gen_kwargs = dict(
            **enc,
            max_new_tokens=args.max_new,
            do_sample=not greedy,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        if not greedy:
            if args.temperature > 0:
                gen_kwargs["temperature"] = args.temperature
            if 0 < args.top_p <= 1:
                gen_kwargs["top_p"] = args.top_p
            if args.top_k > 0:
                gen_kwargs["top_k"] = args.top_k

        gen = model.generate(**gen_kwargs)

        in_lens = enc["attention_mask"].sum(dim=1)
        for i, base_idx in enumerate(range(start, end)):
            prefix_len = int(in_lens[i].item())
            generated = gen[i, prefix_len:].detach().cpu()
            text = tok.decode(generated, skip_special_tokens=True)
            match = TRUE_FALSE_RE.search(text)
            if not match:
                unmatched += 1
                continue
            pred = match.group(1).lower() == "true"
            if pred == answers[base_idx]:
                correct += 1

    matched = total - unmatched
    accuracy = correct / total if total else 0.0
    match_rate = matched / total if total else 0.0
    matched_accuracy = correct / matched if matched else 0.0

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return accuracy, match_rate, matched_accuracy, correct, matched, unmatched


def main():
    args = parse_args()

    prompts, answers = prepare_examples(args.split, args.samples, args.seed)
    print(f"Loaded {len(prompts)} BoolQ examples from split '{args.split}'.")

    results = []
    for mid in args.models:
        print(f"\nEvaluating {mid} ...")
        acc, match_rate, matched_acc, correct, matched, unmatched = evaluate_model(mid, prompts, answers, args)
        results.append((mid, acc, match_rate, matched_acc, correct, matched, unmatched))
        print(
            f"Accuracy: {acc * 100:.2f}% ({correct}/{len(prompts)} correct) | "
            f"Matches: {match_rate * 100:.1f}% ({matched}/{len(prompts)}) | "
            f"Accuracy on matches: {matched_acc * 100:.2f}% | Unmatched: {unmatched}"
        )

    width = max(len(m) for m, *_ in results) + 2
    print("\n=== BoolQ Accuracy Summary ===")
    print(f"{'Model'.ljust(width)} | Acc% | Match% | Acc@Match% | Correct/Total | Matched | Unmatched")
    print("-" * (width + 54))
    total = len(prompts)
    for m, acc, match_rate, matched_acc, corr, matched, unmatched in sorted(results, key=lambda x: x[1], reverse=True):
        print(
            f"{m.ljust(width)} | {acc * 100:5.2f}% | {match_rate * 100:6.2f}% | {matched_acc * 100:9.2f}% | "
            f"{corr}/{total} | {matched} | {unmatched}"
        )


if __name__ == "__main__":
    main()
