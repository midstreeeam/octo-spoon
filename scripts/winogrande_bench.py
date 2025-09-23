#!/usr/bin/env python3
"""Quick Winogrande benchmark for Hugging Face causal LMs.

Each example is formatted as a cloze-style prompt asking the model to reply
with a single letter ("A" or "B"). Predictions are matched with a lightweight
parser that recognizes the first option letter/word and tallies accuracy.

Example:
  python scripts/winogrande_bench.py \
    --models nickypro/tinyllama-42M HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M Qwen/Qwen3-0.6B-Base\
    --config winogrande_xl --split validation --samples 1000 --dtype bfloat16
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import argparse
import random
import re
from typing import Iterable, List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = (
    "Choose the correct option (A or B) to fill in the blank. "
    "Reply with a single token: either the letter A/B or the word from the correct option.\n\n"
    "Sentence: {sentence}\n"
    "Option A: {option1}\n"
    "Option B: {option2}\n"
    "Answer:"
)

LETTER_RE = re.compile(r"([A-Za-z0-9]+)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Winogrande evaluator for HF causal LMs")
    ap.add_argument("--models", nargs="+", required=True, help="HF model ids or local paths")
    ap.add_argument("--config", default="winogrande_xl",
                    help="Winogrande configuration (default: winogrande_xl)")
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"],
                    help="Dataset split to sample from (default: validation)")
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
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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


def prepare_examples(config: str, split: str, samples: int, seed: int) -> Tuple[List[str], List[int], List[Tuple[str, str]]]:
    ds = load_dataset("winogrande", config, split=split)
    total = len(ds)
    if samples > total:
        samples = total

    rng = random.Random(seed)
    indices = rng.sample(range(total), samples)

    prompts: List[str] = []
    answers: List[int] = []
    options: List[Tuple[str, str]] = []

    for idx in indices:
        ex = ds[idx]
        sentence = (ex.get("sentence") or "").strip()
        option1 = (ex.get("option1") or "").strip()
        option2 = (ex.get("option2") or "").strip()
        label = ex.get("answer")
        if isinstance(label, str):
            target = 0 if label.strip() == "1" else 1
        else:
            target = int(label) - 1

        prompt = PROMPT_TEMPLATE.format(sentence=sentence, option1=option1, option2=option2)
        prompts.append(prompt)
        answers.append(target)
        options.append((option1, option2))

    return prompts, answers, options


def parse_prediction(text: str, option_pair: Tuple[str, str]) -> Optional[int]:
    option1, option2 = option_pair
    lowered = text.lower()

    match = LETTER_RE.search(lowered)
    if match:
        token = match.group(1)
        if token in {"a", 'optiona a', "optiona", "option1"}:
            return 0
        if token in {"b", "optiona b", "optionb", "option2"}:
            return 1

        first_opt1 = option1.split()
        first_opt2 = option2.split()
        if first_opt1 and token == first_opt1[0].lower():
            return 0
        if first_opt2 and token == first_opt2[0].lower():
            return 1

    opt1 = option1.lower()
    opt2 = option2.lower()
    found1 = bool(opt1) and opt1 in lowered
    found2 = bool(opt2) and opt2 in lowered
    if found1 ^ found2:
        return 0 if found1 else 1

    return None


def evaluate_model(model_id: str, prompts: List[str], answers: List[int], options: List[Tuple[str, str]], args: argparse.Namespace):
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
            pred = parse_prediction(text, options[base_idx])
            if pred is None:
                unmatched += 1
                continue
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

    prompts, answers, options = prepare_examples(args.config, args.split, args.samples, args.seed)
    print(f"Loaded {len(prompts)} Winogrande examples from {args.config}/{args.split}.")

    results = []
    for mid in args.models:
        print(f"\nEvaluating {mid} ...")
        acc, match_rate, matched_acc, correct, matched, unmatched = evaluate_model(mid, prompts, answers, options, args)
        results.append((mid, acc, match_rate, matched_acc, correct, matched, unmatched))
        print(
            f"Accuracy: {acc * 100:.2f}% ({correct}/{len(prompts)} correct) | "
            f"Matches: {match_rate * 100:.1f}% ({matched}/{len(prompts)}) | "
            f"Accuracy on matches: {matched_acc * 100:.2f}% | Unmatched: {unmatched}"
        )

    width = max(len(m) for m, *_ in results) + 2
    print("\n=== Winogrande Accuracy Summary ===")
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
