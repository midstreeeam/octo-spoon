#!/usr/bin/env python3
# Save-then-judge PLL pipeline to minimize GPU memory pressure.

'''
python pseudo_pll.py \
  --phase both \
  --save run_news.json \
  --models HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-1.7B \
  --judge roberta-large \
  --samples 10 --prompt-min 30 --prompt-max 60 --ref-toks 40

# Judge-only run loading existing generations while scoring with XLM-R
python pseudo_pll.py \
  --phase judge \
  --save run_news.json \
  --judge xlm-roberta-large

# WikiText corpus run scored with RoBERTa
python pseudo_pll.py \
  --phase both \
  --save run_wikitext.json \
  --models nickypro/tinyllama-42M HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-1.7B \
  --judge roberta-large \
  --samples 10 --prompt-min 30 --prompt-max 60 --ref-toks 40 \
  --corpus-ds wikitext --corpus-config wikitext-103-raw-v1 --corpus-split train[:1%]
'''

import os, argparse, random, json, numpy as np, torch
from typing import List, Sequence, Tuple, Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# -------------------------
# Utils
# -------------------------
def set_seed(s: int):
    random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def to_dtype(s: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(s, None)

def empty_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# -------------------------
# Corpus sampling
# -------------------------
def sample_prompts(
    ds_name: str, ds_conf: str, ds_split: str,
    n: int, pmin: int, pmax: int, ref_toks: int, seed: int
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    ds = load_dataset(ds_name, ds_conf, split=ds_split)
    prompts, refs = [], []
    for ex in ds:
        if len(prompts) >= n: break
        text = ex.get("text") or ex.get("content") or ""
        if not text: continue
        words = [w for ln in text.splitlines() for w in ln.strip().split() if w]
        need = pmax + ref_toks + 5
        if len(words) < need: continue
        max_start = len(words) - (pmin + ref_toks)
        if max_start <= 0: continue
        start = rng.randint(0, max_start)
        k = min(rng.randint(pmin, pmax), len(words) - ref_toks - start)
        prompt = " ".join(words[start:start+k]).strip()
        ref    = " ".join(words[start+k:start+k+ref_toks]).strip()
        if len(prompt.split()) >= pmin and len(ref.split()) >= max(5, ref_toks//2):
            prompts.append(prompt); refs.append(ref)
    return prompts, refs

# -------------------------
# Phase 1: Generate & SAVE
# -------------------------
@torch.no_grad()
def generate_model_outputs(
    model_id: str, prompts: Sequence[str], device: str, dtype: str,
    max_new: int, temp: float, top_p: float, batch: int, greedy: bool
) -> List[str]:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    kwargs: Dict = {}
    dt = to_dtype(dtype)
    if dt is not None: kwargs["dtype"] = dt  # modern transformers prefers `dtype`
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device).eval()

    outs: List[str] = []
    for i in range(0, len(prompts), batch):
        batch_prompts = prompts[i:i+batch]
        enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        in_lens = enc["attention_mask"].sum(dim=1)
        gen = model.generate(
            **enc, do_sample=not greedy,
            temperature=None if greedy else temp, top_p=None if greedy else top_p,
            max_new_tokens=max_new, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
        )
        for j in range(len(batch_prompts)):
            cont_ids = gen[j, in_lens[j]:]
            outs.append(tok.decode(cont_ids, skip_special_tokens=True).strip())

    # free LLM asap
    del model
    empty_cuda()
    return outs

def save_run(save_path: str, meta: dict, prompts: List[str], refs: List[str], generations: Dict[str, List[str]]):
    payload = {"meta": meta, "prompts": prompts, "refs": refs, "generations": generations}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# -------------------------
# Phase 2: LOAD & Judge PLL
# -------------------------
@torch.no_grad()
def pll_single(texts: Sequence[str], judge_id: str, device: str, batch_mask: int, max_len: int, dtype: str) -> List[float]:
    tok = AutoTokenizer.from_pretrained(judge_id, use_fast=True)
    dt = to_dtype(dtype)
    mlm_kwargs = {"dtype": dt} if dt is not None else {}
    mlm = AutoModelForMaskedLM.from_pretrained(judge_id, **mlm_kwargs).to(device).eval()

    specials = {tok.pad_token_id, getattr(tok, "cls_token_id", None), getattr(tok, "sep_token_id", None),
                getattr(tok, "bos_token_id", None), getattr(tok, "eos_token_id", None)}
    specials = {x for x in specials if x is not None}
    scores: List[float] = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len)
        ids = enc["input_ids"][0]; attn = enc["attention_mask"][0].bool()
        pos = [i for i, tid in enumerate(ids.tolist()) if attn[i] and tid not in specials]
        if not pos: scores.append(float("inf")); continue
        losses = []
        for k in range(0, len(pos), batch_mask):
            sel = pos[k:k+batch_mask]
            masked = ids.unsqueeze(0).repeat(len(sel), 1)
            for b, p in enumerate(sel): masked[b, p] = tok.mask_token_id
            logits = mlm(input_ids=masked.to(device)).logits
            logp = logits.log_softmax(dim=-1)
            tgt = ids[sel].to(device)
            lp = logp[torch.arange(len(sel), device=device), sel, tgt]
            losses.extend((-lp).tolist())
        scores.append(float(np.mean(losses)))

    # free judge ASAP for next judge (if any)
    del mlm
    empty_cuda()
    return scores

def pll_ensemble(texts: Sequence[str], judges: Sequence[str], device: str, batch_mask: int, max_len: int, judge_dtype: str) -> np.ndarray:
    arrs = [np.array(pll_single(texts, j, device, batch_mask, max_len, judge_dtype), dtype=float) for j in judges]
    return np.mean(arrs, axis=0)

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    # Phase control
    ap.add_argument("--phase", choices=["gen", "judge", "both"], default="both",
                    help="Run generation, judging, or both sequentially.")
    ap.add_argument("--save", default="pll_run.json", help="File to save/load generations.")
    # Models / judges
    ap.add_argument("--models", nargs="+", help="HF CausalLMs (for phase gen).")
    ap.add_argument("--judge",  nargs="+", default=["roberta-large"], help="Masked-LM judge(s) (for phase judge).")
    # Devices & dtypes
    ap.add_argument("--gen-device", default="cuda")
    ap.add_argument("--judge-device", default="cuda")
    ap.add_argument("--gen-dtype",  default="bfloat16", choices=["auto","float16","bfloat16","float32"])
    ap.add_argument("--judge-dtype", default="float16",  choices=["auto","float16","bfloat16","float32"])
    # Generation knobs
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temp",    type=float, default=0.7)
    ap.add_argument("--top-p",   type=float, default=0.9)
    ap.add_argument("--batch-gen", type=int, default=4)
    ap.add_argument("--greedy", action="store_true")
    # PLL knobs
    ap.add_argument("--pll-batch-mask", type=int, default=16)
    ap.add_argument("--pll-maxlen",     type=int, default=256)
    # Corpus
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--prompt-min", type=int, default=30)
    ap.add_argument("--prompt-max", type=int, default=60)
    ap.add_argument("--ref-toks",   type=int, default=40)
    ap.add_argument("--corpus-ds",      default="cc_news")
    ap.add_argument("--corpus-config", default=None)
    ap.add_argument("--corpus-split",   default="train[:2%]")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    # ----------------- Phase 1: Generate & save -----------------
    if args.phase in ("gen", "both"):
        if not args.models:
            raise SystemExit("--models is required for phase=gen/both")
        prompts, refs = sample_prompts(args.corpus_ds, args.corpus_config, args.corpus_split,
                                       args.samples, args.prompt_min, args.prompt_max, args.ref_toks, args.seed)
        print(f"[GEN] Samples: {len(prompts)} | Corpus: {args.corpus_ds}/{args.corpus_config}:{args.corpus_split}")
        generations = {}
        for mid in args.models:
            print(f"[GEN] {mid}")
            outs = generate_model_outputs(
                mid, prompts, args.gen_device, args.gen_dtype,
                args.max_new, args.temp, args.top_p, args.batch_gen, args.greedy
            )
            generations[mid] = outs
        meta = {
            "corpus": f"{args.corpus_ds}/{args.corpus_config}:{args.corpus_split}",
            "seed": args.seed,
            "gen_cfg": dict(max_new=args.max_new, temp=args.temp, top_p=args.top_p, batch_gen=args.batch_gen, greedy=args.greedy),
            "models": args.models,
        }
        save_run(args.save, meta, prompts, refs, generations)
        print(f"[GEN] Saved to {args.save}")

    # ----------------- Phase 2: Load & judge -----------------
    if args.phase in ("judge", "both"):
        with open(args.save, "r", encoding="utf-8") as f:
            payload = json.load(f)
        prompts = payload["prompts"]
        refs    = payload["refs"]
        gens    = payload["generations"]
        print(f"[JUDGE] Loaded {len(prompts)} prompts from {args.save}")
        print(f"[JUDGE] Judges: {args.judge} on {args.judge_device}")

        results = []
        for mid, texts in gens.items():
            pll = pll_ensemble(texts, args.judge, args.judge_device, args.pll_batch_mask, args.pll_maxlen, args.judge_dtype)
            mu, sd = float(pll.mean()), float(pll.std(ddof=1)) if len(pll) > 1 else 0.0
            print(f"[{mid}] mean pNLL: {mu:.4f}  ± {sd:.4f}")
            results.append((mid, mu, sd))

        # Human baseline
        pll_ref = pll_ensemble(refs, args.judge, args.judge_device, args.pll_batch_mask, args.pll_maxlen, args.judge_dtype)
        rmu, rsd = float(pll_ref.mean()), float(pll_ref.std(ddof=1)) if len(pll_ref) > 1 else 0.0
        print(f"[HUMAN] mean pNLL: {rmu:.4f}  ± {rsd:.4f}")

        # Summary
        width = max(len(m) for m,_,_ in results) + 2
        print("\n=== PLL Summary (lower is better) ===")
        print(f"{'Model'.ljust(width)} | mean ± std")
        print("-" * (width + 16))
        for m, mu, sd in sorted(results, key=lambda x: x[1]):
            print(f"{m.ljust(width)} | {mu:.4f} ± {sd:.4f}")
        print(f"{'HUMAN'.ljust(width)} | {rmu:.4f} ± {rsd:.4f}")

if __name__ == "__main__":
    main()
