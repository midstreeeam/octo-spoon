#!/usr/bin/env python3

'''
python pseudo_pll.py \
  --models HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-1.7B \
  --judge roberta-large xlm-roberta-large \
  --samples 20 --prompt-min 30 --prompt-max 120 --ref-toks 40
'''


import os, argparse, random, numpy as np, torch
from typing import List, Sequence, Tuple, Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

def set_seed(s: int):
    random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def to_dtype(s: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(s, None)

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

@torch.no_grad()
def generate(
    model_id: str, prompts: Sequence[str], device: str, dtype: str,
    max_new: int, temp: float, top_p: float, batch: int, greedy: bool
) -> List[str]:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    kwargs: Dict = {}
    dt = to_dtype(dtype)
    if dt is not None: kwargs["torch_dtype"] = dt
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device).eval()

    outs = []
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
    return outs

@torch.no_grad()
def pll_single(texts: Sequence[str], judge_id: str, device: str, batch_mask: int, max_len: int) -> List[float]:
    tok = AutoTokenizer.from_pretrained(judge_id, use_fast=True)
    mlm = AutoModelForMaskedLM.from_pretrained(judge_id).to(device).eval()
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
    return scores  # lower is better

def pll_ensemble(texts: Sequence[str], judges: Sequence[str], device: str, batch_mask: int, max_len: int) -> np.ndarray:
    arrs = [np.array(pll_single(texts, j, device, batch_mask, max_len), dtype=float) for j in judges]
    return np.mean(arrs, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--judge",  nargs="+", default=["roberta-large"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype",  default="bfloat16", choices=["auto","float16","bfloat16","float32"])
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temp",    type=float, default=0.7)
    ap.add_argument("--top-p",   type=float, default=0.9)
    ap.add_argument("--batch-gen", type=int, default=4)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--pll-batch-mask", type=int, default=48)
    ap.add_argument("--pll-maxlen",     type=int, default=256)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--prompt-min", type=int, default=30)
    ap.add_argument("--prompt-max", type=int, default=60)
    ap.add_argument("--ref-toks",   type=int, default=40)
    ap.add_argument("--corpus-ds",      default="wikitext")
    ap.add_argument("--corpus-config",  default="wikitext-103-raw-v1")
    ap.add_argument("--corpus-split",   default="train[:2%]")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    prompts, refs = sample_prompts(args.corpus_ds, args.corpus_config, args.corpus_split,
                                   args.samples, args.prompt_min, args.prompt_max, args.ref_toks, args.seed)
    print(f"Samples: {len(prompts)} | Corpus: {args.corpus_ds}/{args.corpus_config}:{args.corpus_split} | Judges: {args.judge}")

    results = []
    for mid in args.models:
        gens = generate(mid, prompts, args.device, args.dtype, args.max_new, args.temp, args.top_p, args.batch_gen, args.greedy)
        pll = pll_ensemble(gens, args.judge, args.device, args.pll_batch_mask, args.pll_maxlen)
        print(f"[{mid}] mean pNLL: {pll.mean():.4f}  ± {pll.std(ddof=1):.4f}")
        results.append((mid, float(pll.mean()), float(pll.std(ddof=1))))

    # Human baseline on the same prompts
    pll_ref = pll_ensemble(refs, args.judge, args.device, args.pll_batch_mask, args.pll_maxlen)
    print(f"[HUMAN] mean pNLL: {pll_ref.mean():.4f}  ± {pll_ref.std(ddof=1):.4f}")

    # Summary (sorted)
    width = max(len(m) for m,_,_ in results) + 2
    print("\n=== PLL Summary (lower is better) ===")
    print(f"{'Model'.ljust(width)} | mean ± std")
    print("-" * (width + 16))
    for m, mu, sd in sorted(results, key=lambda x: x[1]):
        print(f"{m.ljust(width)} | {mu:.4f} ± {sd:.4f}")
    print(f"{'HUMAN'.ljust(width)} | {pll_ref.mean():.4f} ± {pll_ref.std(ddof=1):.4f}")

if __name__ == "__main__":
    main()
