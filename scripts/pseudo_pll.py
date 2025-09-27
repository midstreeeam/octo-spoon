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
  --corpus-ds wikitext --corpus-config wikitext-103-raw-v1 --corpus-split test[:1%]


# run octo
python scripts/pseudo_pll.py --phase both --save run.json \
    --models HuggingFaceTB/SmolLM2-135M nickypro/tinyllama-42M  \
    --octo-checkpoint checkpoints/tinystory/ar_tinystory_2epoch.pt  \
    --octo-tokenizer gpt2 --prompt-min 30 --prompt-max 60 --ref-toks 40 \
    --samples 50 --corpus-ds wikitext --corpus-config wikitext-103-v1 \
    --corpus-split test[:10%]

# run octo + octodiff
python scripts/pseudo_pll.py --phase both --save run_octodiff.json \
    --models HuggingFaceTB/SmolLM2-135M nickypro/tinyllama-42M \
    --octo-checkpoint checkpoints/tinystory/ar_tinystory_2epoch.pt \
    --octo-tokenizer gpt2 \
    --octodiff-checkpoint checkpoints/octodiff-tinystory/octodiff_story_3ep.pt \
    --octodiff-tokenizer gpt2 --prompt-min 30 --prompt-max 60 --ref-toks 40 \
    --samples 50 --corpus-ds wikitext --corpus-config wikitext-103-v1 \
    --corpus-split test[:10%]
'''

import os, sys, argparse, random, json, numpy as np, torch
from typing import List, Sequence, Tuple, Dict, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from octo_model import inference as octo_inf
    from octo_model.model import OctoForCausalLM
    OCTO_AVAILABLE = True
    OCTO_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional dependency
    OCTO_AVAILABLE = False
    OCTO_IMPORT_ERROR = exc
    octo_inf = None  # type: ignore
    OctoForCausalLM = None  # type: ignore

try:
    from octodiff import inference as octodiff_inf
    from octodiff.model import OctodiffForDiffusionLM
    from octodiff.config import OctodiffConfig
    OCTODIFF_AVAILABLE = True
    OCTODIFF_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - optional dependency
    OCTODIFF_AVAILABLE = False
    OCTODIFF_IMPORT_ERROR = exc
    octodiff_inf = None  # type: ignore
    OctodiffForDiffusionLM = None  # type: ignore
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


def _maybe_adjust_dtype(model: torch.nn.Module, device: torch.device, dtype_str: str) -> torch.nn.Module:
    dt = to_dtype(dtype_str)
    if dt is not None:
        if device.type == "cpu" and dt != torch.float32:
            print(f"[GEN][OCTO] Requested dtype {dtype_str} not supported on CPU; using float32 instead.")
            dt = torch.float32
        return model.to(device=device, dtype=dt)
    return model.to(device)


def _octo_generate_single(
    model: "OctoForCausalLM",
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new: int,
    temp: float,
    top_p: float,
    greedy: bool,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    generated = input_ids
    for _ in range(max_new):
        outputs = model(input_ids=generated, attention_mask=attention_mask)
        logits = outputs["logits"][:, -1, :]

        if greedy:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(temp, 1e-5)
            probs = torch.softmax(logits, dim=-1)

            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative <= top_p
                mask[..., 0] = True
                filtered = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
                denom = filtered.sum(dim=-1, keepdim=True)
                filtered = torch.where(denom > 0, filtered / denom, sorted_probs)
                choice = torch.multinomial(filtered, num_samples=1)
                next_token = torch.gather(sorted_indices, -1, choice)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=-1)
        if attention_mask is not None:
            pad = torch.ones_like(next_token)
            attention_mask = torch.cat([attention_mask, pad], dim=-1)

        eos_id = tokenizer.eos_token_id
        if eos_id is not None and int(next_token.item()) == int(eos_id):
            break

    continuation = generated[0, input_ids.shape[-1]:].detach().cpu()
    return tokenizer.decode(continuation, skip_special_tokens=True).strip()


def generate_octo_outputs(
    prompts: Sequence[str],
    checkpoint: str,
    config_path: Optional[str],
    tokenizer_path: Optional[str],
    device: str,
    dtype: str,
    max_new: int,
    temp: float,
    top_p: float,
    greedy: bool,
) -> List[str]:
    if not OCTO_AVAILABLE:
        detail = f" (import error: {OCTO_IMPORT_ERROR})" if 'OCTO_IMPORT_ERROR' in globals() and OCTO_IMPORT_ERROR else ""
        raise RuntimeError("octo_model package is not available; install or include it to use --octo-checkpoint." + detail)

    ckpt_path = Path(os.path.expanduser(checkpoint)).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Octo checkpoint not found at {ckpt_path}")
    state_dict = octo_inf._load_state_dict(ckpt_path)
    config, cfg_dir = octo_inf._load_config(config_path)
    vocab_size = octo_inf._infer_vocab_size(state_dict)
    if vocab_size is not None and config.vocab_size != vocab_size:
        print(f"[GEN][OCTO] Adjusting vocab_size {config.vocab_size} -> {vocab_size}")
        config.vocab_size = vocab_size
    if config.pad_token_id is None:
        config.pad_token_id = config.eos_token_id

    auto_device = "cuda"
    device_name = device if device not in (None, "auto") else auto_device
    device_obj = torch.device(device_name)
    model = OctoForCausalLM(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[GEN][OCTO] Missing params: {', '.join(missing)}")
    if unexpected:
        print(f"[GEN][OCTO] Unexpected params: {', '.join(unexpected)}")
    model = _maybe_adjust_dtype(model, device_obj, dtype)
    model.eval()

    tok = octo_inf._load_tokenizer(tokenizer_path, cfg_dir, ckpt_path.parent)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    outputs: List[str] = []
    for prompt in prompts:
        outputs.append(
            _octo_generate_single(
                model=model,
                tokenizer=tok,
                prompt=prompt,
                device=device_obj,
                max_new=max_new,
                temp=temp,
                top_p=top_p,
                greedy=greedy,
            )
        )

    del model
    empty_cuda()
    return outputs


def generate_octodiff_outputs(
    prompts: Sequence[str],
    checkpoint: str,
    config_path: Optional[str],
    tokenizer_path: Optional[str],
    device: str,
    dtype: str,
    max_new: int,
    temp: float,
    top_k: int,
    steps: Optional[int],
) -> List[str]:
    if not OCTODIFF_AVAILABLE:
        detail = (
            f" (import error: {OCTODIFF_IMPORT_ERROR})"
            if 'OCTODIFF_IMPORT_ERROR' in globals() and OCTODIFF_IMPORT_ERROR
            else ""
        )
        raise RuntimeError(
            "octodiff package is not available; install or include it to use --octodiff-checkpoint." + detail
        )

    ckpt_path = Path(os.path.expanduser(checkpoint)).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Octodiff checkpoint not found at {ckpt_path}")

    state_dict, meta = octodiff_inf._load_state_dict(ckpt_path)
    cfg_dir: Optional[Path]
    if config_path:
        config, cfg_dir = octodiff_inf._load_config(config_path)
    elif "config" in meta:
        config = OctodiffConfig(**meta["config"])
        cfg_dir = ckpt_path.parent
    else:
        config, cfg_dir = octodiff_inf._load_config(None)

    vocab_size = octodiff_inf._infer_vocab_size(state_dict)
    if vocab_size is not None and config.vocab_size != vocab_size:
        print(f"[GEN][OCTODIFF] Adjusting vocab_size {config.vocab_size} -> {vocab_size}")
        config.vocab_size = vocab_size
    if config.pad_token_id is None:
        config.pad_token_id = config.eos_token_id

    device_name = device if device not in (None, "auto") else ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(device_name)

    model = OctodiffForDiffusionLM(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[GEN][OCTODIFF] Missing params: {', '.join(missing)}")
    if unexpected:
        print(f"[GEN][OCTODIFF] Unexpected params: {', '.join(unexpected)}")
    model = _maybe_adjust_dtype(model, device_obj, dtype)
    model.eval()

    tokenizer = octodiff_inf._load_tokenizer(tokenizer_path, cfg_dir, ckpt_path.parent)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs: List[str] = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device_obj)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device_obj)

        generated = model.sample(
            input_ids,
            attention_mask=attention_mask,
            num_steps=steps,
            max_new_tokens=max_new,
            temperature=temp,
            top_k=top_k,
        )
        continuation = generated[0, input_ids.shape[-1]:].detach().cpu()
        outputs.append(tokenizer.decode(continuation, skip_special_tokens=True).strip())

    del model
    empty_cuda()
    return outputs

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
    # Local Octo model support
    ap.add_argument("--octo-checkpoint", help="Path to a locally trained Octo checkpoint (.pt/.bin/.safetensors).")
    ap.add_argument("--octo-config", help="Optional config path or identifier for the Octo model.")
    ap.add_argument("--octo-tokenizer", help="Tokenizer path/identifier associated with the Octo model.")
    ap.add_argument("--octo-label", default="octo_local", help="Label used when reporting the Octo model.")
    # Local Octodiff model support
    ap.add_argument("--octodiff-checkpoint", help="Path to a locally trained Octodiff checkpoint.")
    ap.add_argument("--octodiff-config", help="Optional config path or identifier for the Octodiff model.")
    ap.add_argument("--octodiff-tokenizer", help="Tokenizer path/identifier associated with the Octodiff model.")
    ap.add_argument("--octodiff-label", default="octodiff_local", help="Label used when reporting the Octodiff model.")
    ap.add_argument("--octodiff-steps", type=int, default=None, help="Inference steps for Octodiff sampling.")
    ap.add_argument("--octodiff-top-k", type=int, default=0, help="Top-k sampling cutoff for Octodiff (0 disables).")
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
        if not any([args.models, args.octo_checkpoint, args.octodiff_checkpoint]):
            raise SystemExit("Provide --models, --octo-checkpoint, and/or --octodiff-checkpoint for phase=gen/both")
        prompts, refs = sample_prompts(args.corpus_ds, args.corpus_config, args.corpus_split,
                                       args.samples, args.prompt_min, args.prompt_max, args.ref_toks, args.seed)
        print(f"[GEN] Samples: {len(prompts)} | Corpus: {args.corpus_ds}/{args.corpus_config}:{args.corpus_split}")
        generations = {}
        listed_models = []
        if args.models:
            for mid in args.models:
                print(f"[GEN] {mid}")
                outs = generate_model_outputs(
                    mid, prompts, args.gen_device, args.gen_dtype,
                    args.max_new, args.temp, args.top_p, args.batch_gen, args.greedy
                )
                generations[mid] = outs
                listed_models.append(mid)
        if args.octo_checkpoint:
            print(f"[GEN] {args.octo_label} (Octo)")
            outs = generate_octo_outputs(
                prompts=prompts,
                checkpoint=args.octo_checkpoint,
                config_path=args.octo_config,
                tokenizer_path=args.octo_tokenizer,
                device=args.gen_device,
                dtype=args.gen_dtype,
                max_new=args.max_new,
                temp=args.temp,
                top_p=args.top_p,
                greedy=args.greedy,
            )
            generations[args.octo_label] = outs
            listed_models.append(args.octo_label)
        if args.octodiff_checkpoint:
            print(f"[GEN] {args.octodiff_label} (Octodiff)")
            outs = generate_octodiff_outputs(
                prompts=prompts,
                checkpoint=args.octodiff_checkpoint,
                config_path=args.octodiff_config,
                tokenizer_path=args.octodiff_tokenizer,
                device=args.gen_device,
                dtype=args.gen_dtype,
                max_new=args.max_new,
                temp=args.temp,
                top_k=args.octodiff_top_k,
                steps=args.octodiff_steps,
            )
            generations[args.octodiff_label] = outs
            listed_models.append(args.octodiff_label)
        meta = {
            "corpus": f"{args.corpus_ds}/{args.corpus_config}:{args.corpus_split}",
            "seed": args.seed,
            "gen_cfg": dict(max_new=args.max_new, temp=args.temp, top_p=args.top_p, batch_gen=args.batch_gen, greedy=args.greedy),
            "models": listed_models,
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
